from datetime import datetime
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import CalD3R_MenD3s_Dataset
from utils.args import args
from utils.utils import pformat_dict
import numpy as np
import os
import models as model_list
import tasks
import wandb
from torch.profiler import profile, record_function, ProfilerActivity #profiler
from utils.utils import compute_class_weights
from utils.transforms import RGB_transf, DEPTH_transf
from torch.utils.tensorboard import SummaryWriter

#!global variables
training_iterations = 0
np.random.seed(13696641)
torch.manual_seed(13696641)

#!gpus setting up: commands in terminal
#   nvidia-smi     to see all available gpus
#   set CUDA_VISIBLE_DEVICES=0   to set the first gpu as the one where to run
#   echo $CUDA_VISIBLE_DEVICES    to check if gpu is set up properly

#!tensorboard visualization command
#   tensorboard --logdir=logs
    
def init_operations():
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    torch.backends.cudnn.benchmark = True
    
    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name


def main():
    global training_iterations
    init_operations()
    args.modality = args.modality

    #!check gpu
    #check0 = os.environ['CUDA_VISIBLE_DEVICES'] 
    #check = torch.cuda.is_available()
    #torch.zeros(1).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #!Mixed precision scaler
    scaler = torch.amp.GradScaler()

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
            val_transf[m] = DEPTH_transf(augment=False)
    #? augmentation is ONLINE: at each epoch, the model sees different augmented versions of the same samples. So it doesn't increase the number of training samples
    #? If OFFLINE augmentation, instead, we multiply the number of training samples by producing some augmented version of each samples.
            
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
    class_weights = compute_class_weights(train_loader).to(device, non_blocking=False)
    
    #?instanciate a different model per modality. 
    models = {}
    logger.info("Instantiating models per modality")
    for m in args.modality:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        models[m] = getattr(model_list, args.models[m].model)()
    
    #?instanciate also the FUSION network
    logger.info('{} Net\tModality: {}'.format(args.models['FUSION'].model, 'FUSION'))
    models['FUSION'] = getattr(model_list, args.models['FUSION'].model)(models['RGB'], models['DEPTH'])
        
    #!Create  EmotionRecognition  object that wraps all the models for each modality    
    emotion_classifier = tasks.EmotionRecognition("emotion-classifier", 
                                                 models, 
                                                 args.batch_size,
                                                 args.total_batch, 
                                                 args.models_dir, 
                                                 scaler, #mixed precision scaler
                                                 class_weights,
                                                 args.models, 
                                                 args=args)
    
    emotion_classifier.grad_clip() #?gradient clipping is applied to the whole model (all the models for each modality)
    emotion_classifier.script() #? script each model per modality
    
    emotion_classifier.load_on_gpu(device)
    
    #!TRAIN and TESTING
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
            
        train(emotion_classifier, train_loader, val_loader, device)

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


def train(emotion_classifier, train_loader, val_loader, device):
    """
    emotion_classifier: Task containing 1 model per modality
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device to use (cpu, gpu)
    """
       
    global training_iterations
    
    #Tensorboard logger
    writer = SummaryWriter('logs')
    
    #? profiler for CPU and GPU (automaticallly updating Tensorboard). profiling the forward+backward passes, loss computation,...  Not keeping trak of memory alloc/dealloc (profile_memory=false)
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=True) 

    data_loader_source = iter(train_loader)
    emotion_classifier.train(True) #? set the model to training mode
    emotion_classifier.zero_grad() #?clear any existing gradient
    
    #?  current_iter  for restoring from a saved model. Otherwise iteration is set to 0.
    iteration = emotion_classifier.current_iter * (args.total_batch // args.batch_size)
    
    #!i: forward+backward of 1 batch of BATCH_SIZE=32. Next iteration will be on next Batch of BATCH_SIZE!!
    #!real_iter: forward+backward of 1 batch of TOTAL_BATCH_SIZE=128 or 256.
    #!epoch: forward and backward of ALL DATASET (If dataset contains 1000 samples and batch size=100, 1 epoch consists of 10 iterations)
    for i in range(iteration, training_iterations): 
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        
        #?PLOT lr and weights for each model
        for m in emotion_classifier.models:
            writer.add_scalar(f'LR for modality/{m}', emotion_classifier.optimizer[m].param_groups[-1]['lr'], real_iter)   
            for name, param in emotion_classifier.models[m].named_parameters(): 
                writer.add_histogram(f'{m}/{name}', param, real_iter)       
            
        #? If the  data_loader_source  iterator is exhausted (i.e., it has iterated over the entire dataset), a  StopIteration  exception is raised. 
        #? The  except StopIteration  block catches this exception and reinitializes the iterator with effectively starting the iteration from the beginning of the dataset again. 
        start_t = datetime.now()
        try:
            source_data, source_label = next(data_loader_source) #?get the next batch of data with next()
        except StopIteration:
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)
        end_t = datetime.now()

        logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
                    f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        #!move data,labels to gpu
        source_label = source_label.to(device, non_blocking=False) #?labels to GPU
        data = {}
        for m in args.modality:
            data[m] = source_data[m].to(device, non_blocking=False) #? data to GPU
        
        # Start profiling
        profiler.start()   
        #? The autocast() context manager allows PyTorch to automatically cast operations inside it to FP16, reducing memory usage and accelerating computations on compatible hardware.
        #? ONLY for forward and loss computation. Backward is automatically done in same precision as forward!!
        with torch.autocast(device_type= ("cuda" if torch.cuda.is_available() else "cpu"), 
                            dtype=torch.float16,
                            enabled=args.amp): 
            logits, _ = emotion_classifier.forward(data)
            emotion_classifier.compute_loss(logits, source_label) #?internally, the scaler, scales the loss to avoid UNDERFLOW of the gradient (too small gradients) since they will  be computed in FP16 (half precision)
        profiler.step() #!update profiler  
        profiler.stop() #!stop profiler   
            
        emotion_classifier.backward(retain_graph=False) 
        emotion_classifier.compute_accuracy(logits, source_label)
        
        #! if TOTAL_BATCH is finished, update weights and zero gradients
        if real_iter.is_integer():  
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                (real_iter, args.train.num_iter, emotion_classifier.loss.val, emotion_classifier.loss.avg,
                    emotion_classifier.accuracy.val[1], emotion_classifier.accuracy.avg[1]))
            #? PLOT TRAINING LOSS
            writer.add_scalar('Loss/train', emotion_classifier.loss.avg, real_iter)
            writer.add_scalar('Accuracy/train', emotion_classifier.accuracy.avg[1], real_iter)
                    
            #emotion_classifier.check_grad() #function that checks norm2 of the gradient (evaluate whether to apply clipping if too large)
            emotion_classifier.step() #step() attribute calls BOTH  optimizer.step()  and, if implemented,  scheduler.step()
            emotion_classifier.zero_grad() #now zero the gradients to avoid accumulating them since this batch has finished
            
        #! every "eval_freq" iterations the validation is done
        if real_iter.is_integer() and real_iter % args.train.eval_freq == 0: 
            val_metrics = validate(emotion_classifier, val_loader, device, int(real_iter))
            
            #?PLOT VALIDATION ACCURACIES
            writer.add_scalar('Accuracy/validation', val_metrics['top1'], int(real_iter))

            if val_metrics['top1'] <= emotion_classifier.best_iter_score:
                logger.info("OLD best accuracy {:.2f}% at iteration {}".format(emotion_classifier.best_iter_score, emotion_classifier.best_iter))
            else:
                logger.info("NEW best accuracy {:.2f}%".format(val_metrics['top1']))
                emotion_classifier.best_iter = real_iter
                emotion_classifier.best_iter_score = val_metrics['top1']

            emotion_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            emotion_classifier.train(True) 
                      
    writer.close()

def validate(emotion_classifier, val_loader, device, it):
    """
    function to validate the model on the test set
    
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    """

    emotion_classifier.reset_acc()
    emotion_classifier.train(False)
    
    logits = {}
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader): #*for each batch in val loader
            
            label = label.to(device, non_blocking=False)
            for m in args.modality:
                data[m] = data[m].to(device, non_blocking=False)
            
            with torch.autocast(device_type= ("cuda" if torch.cuda.is_available() else "cpu"), 
                            dtype=torch.float16,
                            enabled=args.amp): 
                logits, _ = emotion_classifier.forward(data)
            
            emotion_classifier.compute_accuracy(logits, label)

            #?print the validation accuracy at 5 steps: 20% - 40% - 60% - 80% - 100% of validation set
            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          emotion_classifier.accuracy.avg[1], emotion_classifier.accuracy.avg[5]))

        #?at the end print OVERALL validation accuracy on the whole validation set)
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (emotion_classifier.accuracy.avg[1],
                                                                      emotion_classifier.accuracy.avg[5]))
        #?print the PER-CLASS accuracy
        class_accuracies = [(x / y) * 100 for x, y in zip(emotion_classifier.accuracy.correct, emotion_classifier.accuracy.total)]
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(emotion_classifier.accuracy.correct[i_class]),
                                                         int(emotion_classifier.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': emotion_classifier.accuracy.avg[1], 'top5': emotion_classifier.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    #?LOG validation results
    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.name}'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results






if __name__ == '__main__': 
    main()
