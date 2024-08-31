from datetime import datetime
import cv2
import torch
import numpy as np
import os
import wandb

from utils.logger import logger
from utils.utils import pformat_dict
from utils.utils import compute_class_weights, compute_mean_std , plot_confusion_matrix
from utils.Datasets import CalD3RMenD3s_Dataset, BU3DFE_Dataset
from utils.args import args
from utils.utils import GradCAM
import tasks
import models as model_list
from utils.transforms import Transform


from torch.profiler import profile, ProfilerActivity 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


#!global variables
training_iterations = 0
np.random.seed(13696641)
torch.manual_seed(13696641)

#!gpus setting up:  commands in terminal
#   nvidia-smi     to see all available gpus
#   set CUDA_VISIBLE_DEVICES=0    to set the first gpu as the one where to run
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

    #! create datasets
    dataset = {
        'CalD3rMenD3s': CalD3RMenD3s_Dataset,
        'BU3DFE': BU3DFE_Dataset
    }
    global_dataset = dataset[args.dataset.name](name = args.dataset.name, 
                                                modalities = args.modality,
                                                dataset_conf= args.dataset,
                                                transform=None)
    logger.info(f"Global {args.dataset.name} samples: {len(global_dataset)})")
    
    writer_global = SummaryWriter(f'logs/')

    
    #!BFU3DFE cross val
    # fold_accuracies = []  
    # for fold in range(5):
    #     logger.info(f"Fold {fold + 1}")
    #     # Randomly split the dataset
    #     train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.4, random_state=fold) #60%train 40%val
        
    #! CalD3rMenD3s Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)     
    fold_accuracies = []  
    for fold, (train_idx, val_idx) in enumerate(kf.split(global_dataset)): 
        logger.info(f"Fold {fold + 1}")
        
        #Tensorboard logger
        writer_fold = SummaryWriter(f'logs/Fold_{fold + 1}')
        
        train_dataset = dataset[args.dataset.name](name = args.dataset.name, 
                                                    modalities = args.modality,
                                                    dataset_conf= args.dataset,
                                                    transform=None)
        
        val_dataset = dataset[args.dataset.name](name = args.dataset.name, 
                                                    modalities = args.modality,
                                                    dataset_conf= args.dataset,
                                                    transform=None)
        
        
        #? Create Subsets for training and validation for this FOLD
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)
        logger.info(f"Train samples: {len(train_subset)} at fold {fold + 1}")
        logger.info(f"Validation samples: {len(val_subset)} at fold {fold + 1}") 
        
        #! compute mean and std for normalization at this fold
        mean, std = compute_mean_std(train_subset)
        
        # Update the transform for the training and validation datasets (cannot do it before computing mean and std because of augmentations randomness)
        train_dataset.transform = Transform(augment=True, mean=mean, std=std)
        val_dataset.transform = Transform(augment=False, mean=mean, std=std)
        
        
        # Create DataLoader instances
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataset.workers, 
            pin_memory=True, 
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.dataset.workers, 
            pin_memory=True, 
            drop_last=False
        )
        
        #!compute class weights for Weighted Cross Entropy Loss
        class_weights = compute_class_weights(train_loader, norm=False).to(device, non_blocking=True)
    
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
                                                    args.train.lambda_global,
                                                    args.train.lambda_island,
                                                    args=args)
        
        #emotion_classifier.script() #? script each model per modality
        
        emotion_classifier.load_on_gpu(device)
        
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
            
        #* TRAINING
        fold_accuracy = train(emotion_classifier, train_loader, val_loader, fold, device, writer_fold, mean, std)
        fold_accuracies.append(fold_accuracy)
        logger.info(f"Fold {fold + 1} accuracy: {fold_accuracy:.2f}%")
    
    #!final results
    writer_global.add_text('Final results', f"Fold accuracies: {fold_accuracies}")
    average_accuracy = np.mean(fold_accuracies)
    std_dev_accuracy = np.std(fold_accuracies)
    writer_global.add_text('Final results', f"Average Accuracy: {average_accuracy:.2f}%")
    writer_global.add_text('Final results', f"STD Accuracy: {std_dev_accuracy:.2f}%")
    
    writer_global.close()

    
def train(emotion_classifier, train_loader, val_loader, fold, device, writer, mean=None, std=None):
    """
    emotion_classifier: Task containing 1 model per modality
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device to use (cpu, gpu)
    """
       
    global training_iterations
    
    #? profiler for CPU and GPU (automaticallly updating Tensorboard). profiling the forward+backward passes, loss computation,...  Not keeping trak of memory alloc/dealloc (profile_memory=false)
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=True
                    ) if args.profile else None

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
        
        #?PLOT lr and weights for each model (scheduler step is at each BATCH_SIZE iteration)
        for m in emotion_classifier.models:
            writer.add_scalar(f'Fold_{fold}/LR for modality/{m}', emotion_classifier.optimizer[m].param_groups[-1]['lr'], real_iter)   
               
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
        source_label = source_label.to(device, non_blocking=True) #?labels to GPU
        data = {}
        for m in args.modality:
            data[m] = source_data[m].to(device, non_blocking=True) #? data to GPU
        
        # Start profiling
        if profiler:
            profiler.start()   
        #? The autocast() context manager allows PyTorch to automatically cast operations inside it to FP16, reducing memory usage and accelerating computations on compatible hardware.
        #? ONLY for forward and loss computation. Backward is automatically done in same precision as forward!!
        with torch.autocast(device_type= ("cuda" if torch.cuda.is_available() else "cpu"), 
                            dtype=torch.float16,
                            enabled=args.amp): 
            logits, features = emotion_classifier.forward(data)
            emotion_classifier.compute_loss(logits, source_label, features['late']) #?internally, the scaler, scales the loss to avoid UNDERFLOW of the gradient (too small gradients) since they will  be computed in FP16 (half precision)
        if profiler:
            profiler.step() #!update profiler  
            profiler.stop() #!stop profiler   
            
        emotion_classifier.backward(retain_graph=False) 
        emotion_classifier.compute_accuracy(logits, source_label)
        
        #! if TOTAL_BATCH is finished, update weights and zero gradients
        if real_iter.is_integer():  
            logger.info("[%d/%d]\tAvg Loss: %.4f\tAvg Acc Top1: %.2f%%" %
                (real_iter, args.train.num_iter, emotion_classifier.loss.avg, emotion_classifier.accuracy.avg[1]))
            
            #? PLOT TRAINING LOSS and ACCURACY 
            writer.add_scalar(f'Fold_{fold}/Loss/train', emotion_classifier.loss.avg, real_iter)
            writer.add_scalar(f'Fold_{fold}/Accuracy/train', emotion_classifier.accuracy.avg[1], real_iter)
            # #? PLOT WEIGHTS (optimizer step is at TOTAL_BATCH)
            # for m in emotion_classifier.models:
            #     for name, param in emotion_classifier.models[m].named_parameters(): 
            #         writer.add_histogram(f'Fold_{fold}/{m}/{name}', param, real_iter)    
                    
            #emotion_classifier.check_grad() #checks norm2 of the gradient (evaluate whether to apply clipping if too large)
            emotion_classifier.grad_clip() #?gradient clipping is applied to all the models for each modality
            emotion_classifier.step() #step() attribute calls BOTH  optimizer.step()  and, if implemented,  scheduler.step()
            emotion_classifier.zero_grad() #now zero the gradients to avoid accumulating them since this batch has finished
            

        #! every "eval_freq" iterations the validation is done
        if real_iter.is_integer() and real_iter % args.train.eval_freq == 0:  
            val_metrics = validate(emotion_classifier, val_loader, device, int(real_iter))
            
            #?PLOT VALIDATION ACCURACIES
            writer.add_scalar(f'Fold_{fold}/Accuracy/validation', val_metrics['top1'], int(real_iter))

            if val_metrics['top1'] <= emotion_classifier.best_iter_score:
                logger.info("OLD best accuracy {:.2f}% at iteration {}".format(emotion_classifier.best_iter_score, emotion_classifier.best_iter))
            else:
                logger.info("NEW best accuracy {:.2f}%".format(val_metrics['top1']))
                emotion_classifier.best_iter = real_iter
                emotion_classifier.best_iter_score = val_metrics['top1']
                
                #?save the best model if the validation accuracy is improved
                emotion_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            
            #! every  N_val_visualize validations, also visualize features and heatmap
            if real_iter % (args.train.eval_freq*args.N_val_visualize)==0:
                visualize_features(emotion_classifier, val_loader, device, int(real_iter))
                compute_heatmap(emotion_classifier, val_loader, device, int(real_iter), mean, std)
                
            emotion_classifier.train(True) 
               
    #at the end of fold, plot confusion matrix and save it
    confusion_matrix(emotion_classifier, val_loader, device, fold)
    
    #return the best accuracy
    return emotion_classifier.best_iter_score
    
    
def validate(emotion_classifier, val_loader, device, real_iter):
    """
    Validation function
    """

    emotion_classifier.reset_acc()
    emotion_classifier.train(False)
    
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader): #*for each batch in val loader
            
            label = label.to(device, non_blocking=True)
            for m in args.modality:
                data[m] = data[m].to(device, non_blocking=True)
            
            with torch.autocast(device_type= ("cuda" if torch.cuda.is_available() else "cpu"), 
                            dtype=torch.float16,
                            enabled=args.amp): 
                logits, _ = emotion_classifier.forward(data)

            emotion_classifier.compute_accuracy(logits, label)
            
            #?print the validation accuracy at 5 steps: 20% - 40% - 60% - 80% - 100% of validation set
            if (i_val + 1) % (len(val_loader) // 3) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          emotion_classifier.accuracy.avg[1], emotion_classifier.accuracy.avg[5]))
        
        #?at the end print OVERALL validation accuracy on the whole validation set)
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (emotion_classifier.accuracy.avg[1],
                                                                      emotion_classifier.accuracy.avg[5]))
        #?print the PER-CLASS accuracy
        class_accuracies = [(x / y) * 100 if y != 0 else 0 for x, y in zip(emotion_classifier.accuracy.correct, emotion_classifier.accuracy.total)] #? if no samples for a given class, put 0
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
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (real_iter, args.train.num_iter, test_results['top1']))

    return test_results


def confusion_matrix(emotion_classifier, val_loader, device, fold):
    """
    Compute and plot the confusion matrix
    """
    emotion_classifier.train(False)
    confusion_matrix = np.zeros((7, 7))
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):  # For each batch in val loader
            label = label.to(device, non_blocking=True)
            for m in args.modality:
                data[m] = data[m].to(device, non_blocking=True)
            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), 
                                dtype=torch.float16,
                                enabled=args.amp): 
                logits, _ = emotion_classifier.forward(data)
            _, preds = torch.max(logits, 1)
            for t, p in zip(label.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    plot_confusion_matrix(confusion_matrix, fold)
    logger.info(f"Confusion matrix saved at ./Images/confusion_matrix_{fold}.png")
    

def visualize_features(emotion_classifier, val_loader, device, real_iter):
    '''Visualize features and heatmap'''
    
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    val_features = []
    val_labels = []
    
    emotion_classifier.train(False) 
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader): #*for each batch in val loader
            
            label = label.to(device, non_blocking=True)
            for m in args.modality:
                data[m] = data[m].to(device, non_blocking=True)
            
            with torch.autocast(device_type= ("cuda" if torch.cuda.is_available() else "cpu"), 
                            dtype=torch.float16,
                            enabled=args.amp): 
                _, features = emotion_classifier.forward(data)

            #?append, FUSION features for each new batch (for visualization), ordered by label
            val_features.extend(features['late'].cpu().numpy()) 
            val_labels.extend(label.cpu().numpy()) 
                      
    #!plot features for each modality and for the fusion network
    stacked_features = np.vstack(val_features) #? Stack the list of arrays into a single 2D array
    
    tsne = TSNE(n_components=2, random_state=42) #? apply tSNE to reduce to 3D space
    features_2d = tsne.fit_transform(stacked_features)
    
    val_labels = np.array(val_labels)
    cmap = colormaps.get_cmap('tab10')
    for i, label in enumerate(emotions.keys()):
        mask = (val_labels == i)
        plt.scatter(features_2d[:, 0][mask], features_2d[:, 1][mask], color=cmap(i), label=label)
    
    #? plot the features
    #plt.title(f'Features iteration {real_iter}')
    plt.legend()
    plt.savefig(os.path.join('./Images/', f'Features_{real_iter}_iter.png'))
    #plt.show()              
    plt.clf() #clear the plot 
    


def compute_heatmap(emotion_classifier, val_loader, device, real_iter, mean, std):
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    reverse_emotions = {v: k for k, v in emotions.items()}

    emotion_classifier.train(False)
    #!heatmap object
    #feature exractor must produce features retaining some spatial info (must be a conv layer, cannot be an FC)
    gradcam  = GradCAM(emotion_classifier.models['FUSION'], emotion_classifier.models['FUSION'].module.rgb_model.model[2][6]) 
    
    # Dictionary to store one image per class
    class_images = {label: {m: [] for m in args.modality} for label in emotions.values()}
    #!take 1 sample image per class from val loader
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader): 
            label = label.to(device, non_blocking=True)
            for m in args.modality:
                data[m] = data[m].to(device, non_blocking=True)

                # Check if already have an image for each class
                for i in range(len(label)):
                    class_label = label[i].item()
                    if len(class_images[class_label][m]) < 4: #?take 4 images per class
                        class_images[class_label][m].append(data[m][i])
                
            if all(len(data) == 4 for data in class_images.values()):
                break
            
    #! Compute Grad-CAM for a each image class
    for class_label, data in class_images.items():
        if data is not None:
            for i in range(len(data['RGB'])):
                input_data = {'RGB': data['RGB'][i], 'DEPTH': data['DEPTH'][i]}
                heatmap = 1 - gradcam(input_data, class_label)
                img = data['RGB'][i]
                
                # Resize heatmap to the image size and overlay it
                heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
                heatmap_resized = np.uint8(heatmap_resized * 255)
                heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                #recover the original image
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_np = ((img_np * std['RGB']) + mean['RGB'])
                img_np = np.uint8(img_np * 255)
                
                # Overlay the heatmap on the image
                overlay_img = cv2.addWeighted(img_np, 1 - 0.5, heatmap, 0.5, 0)
                
                # Display the result
                plt.imshow(overlay_img)
                plt.title(f'Class: {list(emotions.keys())[list(emotions.values()).index(class_label)]}')
                plt.axis('off')
                plt.savefig(os.path.join('./Images/', f'heatmap_{reverse_emotions[class_label]}_number{i}_iter{real_iter}.png'))
                #plt.show()
                plt.clf()
            
            
if __name__ == '__main__': 
    main()
