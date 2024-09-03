import optuna
import numpy as np
from sklearn.model_selection import KFold
import torch
from utils.args import args
from utils.logger import logger
from utils.Datasets import CalD3RMenD3s_Dataset, BU3DFE_Dataset
import tasks
import models as model_list
from utils.transforms import Transform
from utils.utils import compute_mean_std, compute_class_weights
import os
from tqdm import tqdm


class TqdmCallback:
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials)

    def __call__(self, study, trial):
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()
        
        
training_iterations = 0
def objective(trial):
    global training_iterations
    training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
            
    # Define the hyperparameters to be tuned
    total_batch = trial.suggest_categorical("batch_size", [128, 256, 512])
    fusion_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    backbone_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    global_lambda = trial.suggest_float("global_lambda", 1e-5, 1e-1, log=True)
    island_lambda = trial.suggest_float("island_lambda", 0.1, 100, log=True)
    
    #Overwrite the hyperparameters
    args.total_batch = total_batch
    args.train.weight_decay = weight_decay
    args.models['FUSION'].lr = fusion_lr
    args.models['RGB'].lr = backbone_lr
    args.models['DEPTH'].lr = backbone_lr
    args.global_lambda = global_lambda
    args.island_lambda = island_lambda
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #!Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    
    # Recreate your dataset and dataloaders if necessary
    dataset = {
        'CalD3rMenD3s': CalD3RMenD3s_Dataset,
        'BU3DFE': BU3DFE_Dataset
    }
    global_dataset = dataset[args.dataset.name](name=args.dataset.name, 
                                                modalities=args.modality,
                                                dataset_conf=args.dataset,
                                                transform=None)
    
    # Implement your K-Fold Cross Validation and Training Logic
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(global_dataset)): 
        train_dataset = dataset[args.dataset.name](name=args.dataset.name, 
                                                    modalities=args.modality,
                                                    dataset_conf=args.dataset,
                                                    transform=None)
        val_dataset = dataset[args.dataset.name](name=args.dataset.name, 
                                                 modalities=args.modality,
                                                 dataset_conf=args.dataset,
                                                 transform=None)
        
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)
        
        mean, std = compute_mean_std(train_subset)
        
        train_dataset.transform = Transform(augment=True, mean=mean, std=std)
        val_dataset.transform = Transform(augment=False, mean=mean, std=std)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        #!compute class weights for Weighted Cross Entropy Loss
        class_weights = compute_class_weights(train_loader, norm=False).to(device, non_blocking=True)
        
        # Initialize your model and emotion classifier
        models = {}
        for m in args.modality:
            models[m] = getattr(model_list, args.models[m].model)()
        models['FUSION'] = getattr(model_list, args.models['FUSION'].model)(models['RGB'], models['DEPTH'])
        
        emotion_classifier = tasks.EmotionRecognition("emotion-classifier", 
                                                    models, 
                                                    args.batch_size,
                                                    args.total_batch, 
                                                    args.models_dir, 
                                                    scaler, #mixed precision scaler
                                                    class_weights,
                                                    args.models, 
                                                    args.global_lambda,
                                                    args.island_lambda,
                                                    args=args)
        
        emotion_classifier.load_on_gpu(device)
        fold_accuracy = train(emotion_classifier, train_loader, val_loader, fold, device, mean, std)
        fold_accuracies.append(fold_accuracy)
    
        # Calculate the average accuracy across folds
        average_accuracy = np.mean(fold_accuracies)
        
        return average_accuracy #ONLY FIRST FOLD IS RETURNED


def train(emotion_classifier, train_loader, val_loader, fold, device, mean=None, std=None):

    global training_iterations
    
    data_loader_source = iter(train_loader)
    emotion_classifier.train(True) #? set the model to training mode
    emotion_classifier.zero_grad() #?clear any existing gradient
    
    #?  current_iter  for restoring from a saved model. Otherwise iteration is set to 0.
    iteration = emotion_classifier.current_iter * (args.total_batch // args.batch_size)
    
    for i in range(iteration, training_iterations): 
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        
        try:
            source_data, source_label = next(data_loader_source) #?get the next batch of data with next()
        except StopIteration:
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)

        #!move data,labels to gpu
        source_label = source_label.to(device, non_blocking=True) #?labels to GPU
        data = {}
        for m in args.modality:
            data[m] = source_data[m].to(device, non_blocking=True) #? data to GPU
        
        logits, features = emotion_classifier.forward(data)
        emotion_classifier.compute_loss(logits, source_label, features['late']) 
        emotion_classifier.backward(retain_graph=False) 
        emotion_classifier.compute_accuracy(logits, source_label)
        
        #! if TOTAL_BATCH is finished, update weights and zero gradients
        if real_iter.is_integer():  
            emotion_classifier.grad_clip() #?gradient clipping is applied to all the models for each modality
            emotion_classifier.step() #step() attribute calls BOTH  optimizer.step()  and, if implemented,  scheduler.step()
            emotion_classifier.zero_grad() #now zero the gradients to avoid accumulating them since this batch has finished
            
        #! every "eval_freq" iterations the validation is done
        if real_iter.is_integer() and real_iter % args.train.eval_freq == 0:  
            val_metrics = validate(emotion_classifier, val_loader, device, int(real_iter))
            emotion_classifier.best_iter = real_iter
            emotion_classifier.best_iter_score = val_metrics['top1']
             
            emotion_classifier.train(True) 
               
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

if __name__ == "__main__":
    n_trials = 50
    tqdm_callback = TqdmCallback(n_trials)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
    
    # Close the progress bar once done
    tqdm_callback.close()
    
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)