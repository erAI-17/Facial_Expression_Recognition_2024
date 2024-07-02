from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger
from typing import Dict, Tuple
from utils.Losses import FocalLoss, CenterLoss


class EmotionRecognition(tasks.Task, ABC):
    def __init__(self, 
                 name: str, 
                 task_models: Dict[str, torch.nn.Module], 
                 batch_size: int, 
                 total_batch: int, 
                 models_dir: str, 
                 scaler,
                 class_weights: torch.FloatTensor, 
                 model_args: Dict[str, float], 
                 args, **kwargs) -> None:
        
        """
        Parameters
        ----------
        name : str
            name of the task e.g. emotion_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        scaler : 
            scaler object fro mixed precision
        class_weights: float tensor
            weights for each class to use for Weighted cross entropy loss
        model_args : Dict[str, float]
            model-specific arguments
        """
        
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args
        
        #!self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5))
        self.loss = utils.AverageMeter()
        
        #!scaler for mixed precision 
        self.scaler = scaler
        
        #! CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                    reduce=None, reduction='none')
        #!Weighted CEL
        #self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, size_average=None, ignore_index=-100,
        #                                           reduce=None, reduction='none')
        
        #!Focal Loss #dynamically scales the loss for each sample based on the prediction confidence.
        #self.criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
        
        #!CEL+Center Loss 
        #self.criterion = CEL_CL_Loss(alpha=1, gamma=2, reduction='mean')
        
        # Initialize the model parameters and the optimizer
        optim_params = {}
        self.optimizer = {}
        self.scheduler = {}
        for m in self.task_models:
            #?select only parameters of the model that have requires_grad == True. If they have requires_grad == False it means they should not be
            #? includeed in gradient computation and NOT be updated because of PRE-TRAINING FREEZING
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            
            #? optim_params[m] : The parameters of the model  m  that require gradients. 
            #? model_args[m].lr : Initial learning rate for the optimizer
            #? weight_decay : The weight decay (L2 penalty) for the optimizer. 
            #!ADAM
            self.optimizer[m] = torch.optim.Adam(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay)
            # Use a learning rate scheduler to decrease the learning rate over time
            self.scheduler[m] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[m], T_max=args.train.num_iter, eta_min=1e-6)
            
            #!SGD with momentum
            # self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
            #                                     weight_decay=model_args[m].weight_decay,
            #                                     momentum=model_args[m].sgd_momentum)
            # #Use a learning rate scheduler to decrease the learning rate over time
            # self.scheduler[m] = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[m], T_max=args.train.num_iter, eta_min=1e-6)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor] ->{'RGB':tensor[32, 3, 224,224], 'DEPTH':tensor[32, 1, 224,224] } #* [BATCH_SIZE, C , H, W]
            a dictionary that stores the input data for each modality. 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        #!train all modalities networks TOGETHER passing data to FUSION network
        logits = {}
        features = {}
        logits, feat = self.task_models['FUSION'](data, **kwargs) #logits [32,7]
        #? return features to PLOT which are more discriminative (try different loss functions)
        for i_m, m in enumerate(self.modalities):
            if i_m == 0: #initially set up an empty dictionary for each modality to store corresponding features
                for k in feat.keys():
                    features[k] = {} 
            
            for k in feat.keys(): #for each level of feature extraction (early-mid-late), save the features extracted
                features[k][m] = feat[k]

        return logits, features


    def compute_loss(self, logits: torch.Tensor, label: torch.Tensor):
        """Compute the classification loss.

        Parameters
        ----------
        logits : final logits
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        loss = self.criterion(logits, label) #?the criterion is defined with "reduction=None", so the loss function returns a tensor containing the loss for each individual sample in the batch,
                                             #? rather than averaging or summing the losses. LATER, before backward pass, the loss is averaged over the batch size.
        
        #? Update the loss value, weighting it by the ratio of the batch size to the total batch size (for gradient accumulation)
        self.loss.update(torch.mean(loss) / (self.total_batch / self.batch_size), self.batch_size)


    def compute_accuracy(self, logits: torch.Tensor, label: torch.Tensor):
        """Compute the classification accuracy.

        Parameters
        ----------
        logits : final logits
        label : torch.Tensor
            ground truth
        """
        self.accuracy.update(logits, label)

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        self.loss.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        #!!if using mixed precision autocast() in training
        # for m in self.modalities:
        #     # Perform the step with the optimizer and scaler
        #     self.scaler.step(self.optimizer[m]) #? instead of simple self.optimizer[m].step() , YOU NEED TO SCALE THE GRADIENTS BACK to FP32
            
        #     # Perform the step with the scheduler (if any)
        #     if self.scheduler[m] is not None:  
        #         self.scheduler[m].step()
        
        # # Update the scaler for the next iteration        
        # self.scaler.update()
        
         #!!if using mixed precision autocast() in training
        for m in self.modalities:
            # Perform the step with the optimizer and scaler
            self.optimizer[m].step() #? instead of simple self.optimizer[m].step() , YOU NEED TO SCALE THE GRADIENTS BACK to FP32
            
            # Perform the step with the scheduler (if any)
            if self.scheduler[m] is not None:  
                self.scheduler[m].step()
                
        # Reset loss and accuracy tracking
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph (for example if using multiple different losses)

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        
        print('CURRENT LOSS', self.loss.val )
        print('SCALED LOSS', self.scaler.scale(self.loss.val) )
        
        #!if using mixed precision autocast() in training
        #self.loss.val is the loss value over a mini-batch 32 (NOT the accumulated loss over the 4 32 batches inside effective batch 128)
        
        #!if NOT using mixed precision:
        self.loss.val.backward(retain_graph=retain_graph)
        
    def wandb_log(self):
            """Log the current loss and top1/top5 accuracies to wandb."""
            logs = {
                'loss verb': self.loss.val, 
                'top1-accuracy': self.accuracy.avg[1],
                'top5-accuracy': self.accuracy.avg[5]
            }

            # Log the learning rate, separately for each modality.
            for m in self.modalities:
                logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
            wandb.log(logs)