from datetime import datetime
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import CalD3R_MenD3s_Dataset
from utils.args import args
from utils.utils import pformat_dict
import matplotlib.pyplot as plt
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb
from utils.utils import compute_class_weights
from utils.transforms import RGB_transf, DEPTH_transf

# global variables among training functions
training_iterations = 0
np.random.seed(13696641)
torch.manual_seed(13696641)

#!gpus setting up: commands in terminal
#   nvidia-smi     to see all available gpus
#   set CUDA_VISIBLE_DEVICES=0   to set the first gpu as the one where to run
#   echo $CUDA_VISIBLE_DEVICES    to check if gpu is set up properly
    
def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name


def main():
    global training_iterations
    init_operations()
    args.modality = args.modality

    # recover num_classes, valid paths, domains, 
    num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
    
    #!check gpu
    #check0 = os.environ['CUDA_VISIBLE_DEVICES'] 
    check = torch.cuda.is_available()
    torch.zeros(1).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #!TRANSFORMATIONS and AUGMENTATION for TRAINING samples, 
    #!ONLY TRANSFORMATION (NO AUGMENTATION) for VALIDATION/TEST samples
    train_transf = {}
    val_transf = {}
    for m in args.modality:
        if m == 'RGB':
            train_transf[m] = RGB_transf(augment=True)
            val_transf[m] = RGB_transf(augment=False)
        if m == 'DEPTH':
            train_transf[m] = DEPTH_transf(augment=True)
            val_transf[m] = DEPTH_transf(augment=True)
    #? augmentation is ONLINE: at each epoch, the model sees different augmented versions of the same samples. So it doesn't increase the number of training samples
    #? If OFFLINE augmentation, instead, we multiply the number of trining samples by producing some augmented version of each samples.
            
    train_loader = torch.utils.data.DataLoader(CalD3R_MenD3s_Dataset(args.dataset.name,
                                                                        args.modality,
                                                                        'train', 
                                                                        args.dataset,
                                                                        train_transf,
                                                                        additional_info=False),
                                                batch_size=args.batch_size, #small BATCH_SIZE
                                                shuffle=True,
                                                num_workers=args.dataset.workers, 
                                                pin_memory=True, 
                                                drop_last=True)

    val_loader = torch.utils.data.DataLoader(CalD3R_MenD3s_Dataset(args.dataset.name,
                                                                    args.modality,
                                                                    'val', 
                                                                    args.dataset,
                                                                    val_transf,
                                                                    additional_info=False),
                                                batch_size=args.batch_size, 
                                                shuffle=False,
                                                num_workers=args.dataset.workers, 
                                                pin_memory=True, 
                                                drop_last=False)
    
    #!compute class weights for Weighted Cross Entropy Loss
    class_weights = compute_class_weights(train_loader).to(device)
    
    #!Create  EmotionRecognition  object that wraps all the models for each modality
    
    #?if FUSING modalities, ONLY instanciate the fusion network. Else, instanciate a different model per modality and train them separately for logits fusion
    if args.fusion_modalities == True:
        logger.info("Instantiating model: %s", args.models['FUSION'].model)
        
        fusion_model = getattr(model_list, args.models['FUSION'].model)()
        models = {'FUSION':fusion_model }
    else:
        models = {}
        for m in args.modality:
            logger.info("Instantiating models per modality")
            logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
            models[m] = getattr(model_list, args.models[m].model)()
        

    emotion_classifier = tasks.EmotionRecognition("emotion-classifier", 
                                                 models, 
                                                 args.batch_size,
                                                 args.total_batch, 
                                                 args.models_dir, 
                                                 class_weights,
                                                 args.models, 
                                                 args=args)
    emotion_classifier.load_on_gpu(device)
    
    #!handle TRAIN and TESTING
    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            emotion_classifier.load_last_model(args.resume_from)
        
        #* USE GRADIENT ACCUMULATION (the batches are devided into smaller batches with gradient accumulation).
        #* RECALL that larger batch sizes lead to faster convergence because the estimated gradients are more accurate (more similar to the one that would be computed over the whole dataset)
        #* and less affected by noise. So we want large batch sizes BUT we may have some memory constraints, so we use GRADIENT ACCUMULATION
      
        #* TOTAL_BATCH (128) -> 4* BATCH_SIZE (32)
        #* There are 4 BATCH_SIZE inside TOTAL_BATCH each of which must be iterated (forward+backward) "args.train.num_iter" times,
        #* so total number of iterations done over the whole dataset (since we are using smaller batches BATCH_SIZE) is
        training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
            
        train(emotion_classifier, train_loader, val_loader, device, num_classes)

    #!TEST
    elif args.action == "test":
        if args.resume_from is not None:
            emotion_classifier.load_last_model(args.resume_from)
            
        test_loader = torch.utils.data.DataLoader(CalD3R_MenD3s_Dataset(args.dataset.name,
                                                                        args.modality,
                                                                        'test', 
                                                                        val_transf,
                                                                        args.dataset),
                                                 batch_size=args.batch_size, 
                                                 shuffle=False,
                                                 num_workers=args.dataset.workers, 
                                                 pin_memory=True, 
                                                 drop_last=False)

        validate(emotion_classifier, test_loader, device, emotion_classifier.current_iter)


def train(emotion_classifier, train_loader, val_loader, device, num_classes):
    """
    function to train 1 model for modality on the training set
    
    emotion_classifier: Task containing the model to be trained for each modality
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device to use (cpu, gpu)
    num_classes: int, number of classes in the classification problem
    """
       
    global training_iterations

    data_loader_source = iter(train_loader)
    emotion_classifier.train(True) #? set the model to training mode
    emotion_classifier.zero_grad() #?clear any existing gradient
    
    #*current_iter is just for restoring from a saved run. Otherwise iteration is set to 0.
    iteration = emotion_classifier.current_iter * (args.total_batch // args.batch_size)

    
    training_losses = []
    validation_accuracies = []
    try: #? try-finallly construct to plot training losses and validation accuracies even if you stop earlier the function (Keyboardinterrupt exception raises)
        
        #*iteration: forward and backward of 1 batch of BATCH_SIZE. Next iteration will be on next Batch of BATCH_SIZE!!
        #*epoch: forward and backward of ALL DATASET (If dataset contains 1000 samples and batch size= 100, 1 epoch consists of 10 iterations)
        for i in range(iteration, training_iterations): #ITERATIONS on batches (of BATCH_SIZE) 
            real_iter = (i + 1) / (args.total_batch // args.batch_size)
            
            
            #? reduce learning rate from ad hoc function (deviding by 10) 
            #if real_iter == args.train.lr_steps:
            #    emotion_classifier.reduce_learning_rate() 
            #?otherwise use a lr scheduler defined in task
            learning_rates = []
            if args.fusion_modalities == True:                
                current_lr = emotion_classifier.optimizer['FUSION'].param_groups[-1]['lr']
                learning_rates.append((current_lr, real_iter))
                logger.info(f"Current learning rate: {current_lr}")
            else:
                for m in args.modality:
                    current_lr = emotion_classifier.optimizer[m].param_groups[-1]['lr']
                    learning_rates.append((current_lr, real_iter))
                    logger.info(f"Current learning rate: {current_lr}")
                    
                
            #*we reason in terms of ITERATIONS on batches (of BATCH_SIZE) not EPOCHS!!
            #? If the  data_loader_source  iterator is exhausted (i.e., it has iterated over the entire dataset), a  StopIteration  exception is raised. 
            #? The  except StopIteration  block catches this exception and reinitializes the iterator with effectively starting the iteration from the beginning of the dataset again. 
            start_t = datetime.now()
            try:
                source_data, source_label = next(data_loader_source) #*get the next batch of data with next()!
            except StopIteration:
                data_loader_source = iter(train_loader)
                source_data, source_label = next(data_loader_source)
            end_t = datetime.now()

            logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
                        f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

            source_label = source_label.to(device) #?labels to GPU
            data = {}
            #?if FUSING modalities, data dictionary contains RGB and DEPTH for FUSION network. 
            if args.fusion_modalities == True:
                data['FUSION'] = {m: source_data[m].to(device) for m in args.modality}
            else: #? else data dictionary contains SEPARATE modality data for each modality (they will be passed into different models)
                for m in args.modality:
                    data[m] = source_data[m].to(device) #? data to GPU
                    
            logits, _ = emotion_classifier.forward(data)
            emotion_classifier.compute_loss(logits, source_label, loss_weight=1)
            emotion_classifier.backward(retain_graph=False)
            emotion_classifier.compute_accuracy(logits, source_label)
                    
            #? update weights and zero gradients if TOTAL_BATCH is finished!!
            #? also print the training loss MEAN over the 4 32batches inside the TOTAL_BATCH
            if real_iter.is_integer(): 
                logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                            (real_iter, args.train.num_iter, emotion_classifier.loss.val, emotion_classifier.loss.avg,
                            emotion_classifier.accuracy.val[1], emotion_classifier.accuracy.avg[1]))

                training_losses.append(((emotion_classifier.loss.avg.item()), real_iter)) #? PLOT TRAINING LOSS
                emotion_classifier.check_grad()
                emotion_classifier.step() #step() attribute calls BOTH  optimizer.step()  and, if implemented,  scheduler.step()
                emotion_classifier.zero_grad()
                
        
            #? every "eval_freq" iterations the validation is done
            if real_iter.is_integer() and real_iter % args.train.eval_freq == 0:
                val_metrics = validate(emotion_classifier, val_loader, device, int(real_iter))
                validation_accuracies.append((val_metrics['top1'], real_iter)) #?PLOT VALIDATION ACCURACIES

                if val_metrics['top1'] <= emotion_classifier.best_iter_score:
                    logger.info("OLD best accuracy {:.2f}% at iteration {}".format(emotion_classifier.best_iter_score, emotion_classifier.best_iter))
                else:
                    logger.info("NEW best accuracy {:.2f}%".format(val_metrics['top1']))
                    emotion_classifier.best_iter = real_iter
                    emotion_classifier.best_iter_score = val_metrics['top1']

                emotion_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
                emotion_classifier.train(True)  

    finally:
        # Plot the training loss and validation accuracy
        plt.figure(figsize=(12, 5))
        
        #? Plot Training Loss along with learning rates at each iteration
        if training_losses:
            # Unzip the training_losses into two lists: one for loss values and one for iterations
            losses, real_iter = zip(*training_losses)
            lrs, _ = zip(*learning_rates)
            plt.subplot(1, 2, 1)
            plt.plot(real_iter, losses, label='Training Loss')
            plt.plot(real_iter, lrs, label='learning Rate')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            #plt.savefig('training_loss.png')  # Save the training loss plot

        #? Plot Validation Accuracy along with learning rates at each iteration
        if validation_accuracies:
            # Unzip the training_losses into two lists: one for loss values and one for iterations
            val_acc, real_iter = zip(*validation_accuracies)
            plt.subplot(1, 2, 2)
            plt.plot(real_iter, val_acc, label='Validation Accuracy')
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig('./Images/')
        plt.show()


def validate(model, val_loader, device, it):
    """
    function to validate the model on the test set
    
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    """

    model.reset_acc()
    model.train(False)
    
    logits = {}
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader): #*for each batch in val loader
            
            label = label.to(device)
            #?if FUSING modalities, data dictionary contains RGB and DEPTH for FUSION network. 
            if args.fusion_modalities == True:
                data['FUSION'] = {m: data[m].to(device) for m in args.modality} #a dictionary
            else: #? else data dictionary contains SEPARATE modality data for each modality (they will be passed into different models)
                for m in args.modality:
                    data[m] = data[m].to(device) #? data to GPU
            
            logits, _ = model.forward(data)
            model.compute_accuracy(logits, label)

            #?print the validation accuracy at 5 steps: 20% - 40% - 60% - 80% - 100% of validation set
            #? (only look at the last one if you're only interested on OVERALL validation accuracy on the whole validation set)
            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        #?at the end print OVERALL validation accuracy on the whole validation set)
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        #? at the end print the PER-CLASS accuracy
        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.name}'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results






if __name__ == '__main__':
    main()
