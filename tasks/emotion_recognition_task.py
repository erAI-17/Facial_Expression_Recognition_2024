from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger

from typing import Dict, Tuple


class EmotionRecognition(tasks.Task, ABC):
    def __init__(self, 
                 name: str, 
                 task_models: Dict[str, torch.nn.Module], 
                 batch_size: int, 
                 total_batch: int, 
                 models_dir: str, 
                 num_classes: int, 
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
        num_classes : int
            number of labels in the classification task
        model_args : Dict[str, float]
            model-specific arguments
        """
        
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args
        
        #!self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5))
        self.loss = utils.AverageMeter()
        
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        
        # Initialize the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

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
        logits = {}
        features = {}
        for i_m, m in enumerate(self.modalities):
            logits[m], feat = self.task_models[m](data[m], **kwargs) #logits [32,7]
            
            if i_m == 0: #initially set up an empty dictionary for each modality to store corresponding features
                for k in feat.keys():
                    features[k] = {} #dictionary of dictionaries
            
            for k in feat.keys(): #save features for each modality inside dictionary features
                features[k][m] = feat[k]

        return logits, features


    def compute_loss(self, logits: Dict[str, torch.Tensor], label: torch.Tensor, loss_weight: float=1.0):
        """Fuse the logits from different modalities and compute the classification loss.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        #!modality logits fusion for loss 
        fused_logits = reduce(lambda x, y: x + y, logits.values()) #[32,7] #?creates 1 array of logits by summing ALL arrays of logits from different modalities 
        loss = self.criterion(fused_logits, label)  # [32]
        
        #? Update the loss value, weighting it by the ratio of the batch size to the total batch size (for gradient accumulation)
        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)


    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        #!modality logits fusion for accuracy
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(fused_logits, label)


    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

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
        super().step() #?calls step method of the parents class, which refers to the step method of the optimizer for that modality network
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
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