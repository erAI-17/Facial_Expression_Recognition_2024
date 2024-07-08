from abc import ABC
import torch
from utils import utils
import wandb
import tasks
from typing import Dict, Tuple
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from utils.losses import FocalLoss #, CenterLoss
from utils.args import args

class EmotionRecognition(tasks.Task, ABC):
    def __init__(self, 
                 name: str, 
                 models: Dict[str, torch.nn.Module], 
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
        models : Dict[str, torch.nn.Module]
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
        
        super().__init__(name, models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args
        self.class_weights = class_weights
        
        #!self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5))
        self.loss = utils.AverageMeter()
        
        #!scaler for mixed precision 
        self.scaler = scaler
        
        #! CrossEntropyLoss (already reduce the loss over batch samples with MEAN to not make it dependent on the batch size)
        #self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        
        #!Weighted CEL
        #self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        
        #!Focal Loss #dynamically scales the loss for each sample based on the prediction confidence.
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=2, reduction='mean')
        
        #!CEL+Center Loss 
        #self.criterion = CEL_CL_Loss(alpha=1, gamma=2, reduction='mean')
        
        # Initialize the model parameters and the optimizer
        optim_params = {}
        self.optimizer = {}
        self.scheduler = {}
        for m in self.models:
            #?select only parameters of the model that have requires_grad == True. If they have requires_grad == False it means they should not be
            #? includeed in gradient computation and NOT be updated because of PRE-TRAINING FREEZING
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.models[m].parameters())
            
            #? optim_params[m] : The parameters of the model  m  that require gradients. 
            #? model_args[m].lr : Initial learning rate for the optimizer
            #? weight_decay : The weight decay (L2 penalty) for the optimizer. 
            #!ADAM
            self.optimizer[m] = torch.optim.Adam(optim_params[m], model_args[m].lr, weight_decay=model_args[m].weight_decay)
            #self.optimizer[m] = torch.optim.AdamW(optim_params[m], model_args[m].lr, weight_decay=model_args[m].weight_decay)
            
            #!LR schedulers
            #?warm up schedule
            warmup_iters = 3  # Number of iterations for warm-up
            warmup_start_lr = 1e-7  # Starting learning rate at the beginning of warm-up
            #? start_factor: The factor by which the learning rate is multiplied at the beginning of the warm-up phase. 
            #?So you are actually starting from 1e-7, going linearly to "model_args[m].lr" (over 20 iterations) and then cosine annealing start
            #self.Warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer[m], start_factor=warmup_start_lr/model_args[m].lr, total_iters=warmup_iters)

            #?Cosine Annealing 
            #self.CosineAnnealing = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer[m], T_max=args.train.num_iter - warmup_iters, eta_min=1e-6)
            
            #? CosineAnnealingWarmRestarts scheduler
            #self.CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts(self.optimizer[m], T_0=10, T_mult=2, eta_min=1e-6) #T_0= every 10 epochs, then every 20 epochs, 40 ...

            #?milestones parameter receives [warmup_iters] to specify that the transition from the warm-up scheduler to the cosine annealing scheduler should occur after warmup_iters iterations.
            #self.scheduler[m] = torch.optim.lr_scheduler.SequentialLR(self.optimizer[m], schedulers=[self.Warmup_scheduler, self.CosineAnnealing], milestones=[warmup_iters])

            #?OneCycleLR scheduler
            self.scheduler[m] = OneCycleLR(self.optimizer[m], max_lr=model_args[m].lr, total_steps=args.train.num_iter, anneal_strategy='cos')
            
    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor] ->{'RGB':tensor[32, 3, 224,224], 'DEPTH':tensor[32, 1, 224,224] }
            a dictionary that stores the input data for each modality. 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        #!train all modalities networks TOGETHER passing data to FUSION network
        logits = {}
        features = {}
        logits, feat = self.models['FUSION'](data['RGB'],data['DEPTH'], **kwargs) #logits [32,7]
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
        loss = self.criterion(logits, label) #? IF the criterion is defined with "reduction=None", so the loss function returns a tensor containing the loss for each individual sample in the batch,
                                             #? rather than averaging or summing the losses. LATER, before backward pass, the loss is averaged over the batch size.
        
        #? Update the loss value, weighting it by the ratio of the batch size to the total batch size (for gradient accumulation)
        self.loss.update(loss / (self.total_batch / self.batch_size), self.batch_size)


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
        """This method performs an optimization step and resets both the loss and the accuracy.
        """
        for m in self.modalities:
            if args.amp:  #!!if using mixed precision autocast() in training
                # Perform the step with the optimizer and scaler
                self.scaler.step(self.optimizer[m]) #? instead of simple self.optimizer[m].step() , YOU NEED TO SCALE THE GRADIENTS BACK TO THE ORIGINAL VALUES before updating the model parameters
            else:
                self.optimizer[m].step()
            # Perform the step with the scheduler (if any)
            if self.scheduler[m] is not None:  
                self.scheduler[m].step()
            
        if args.amp: #! Update the scaler for the next iteration        
            self.scaler.update()
                    
        # Reset loss and accuracy tracking
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph (for example if using multiple different losses)
        """
        if args.amp: #!if using mixed precision autocast() in training
            print("LOSS: ", self.loss.val)
            print("scaled LOSS: ", self.scaler.scale(self.loss.val))
            self.scaler.scale(self.loss.val).backward(retain_graph=retain_graph)
        else:
            self.loss.val.backward(retain_graph=retain_graph)
    
    def zero_grad(self):
        """Reset the gradient when gradient accumulation is finished."""
        for m in self.modalities:
            self.optimizer[m].zero_grad(set_to_none=True)
            
    def grad_clip(self):
        """Clip the gradients to avoid exploding gradients."""
        for m in self.modalities:
            torch.nn.utils.clip_grad_norm_(self.models[m].parameters(), args.train.max_grad_norm)
            
    def script(self):
        """Script ONLY the FUSION model containing the feature extraction models"""
        self.models['FUSION'] = torch.jit.script(self.models['FUSION'])
        
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