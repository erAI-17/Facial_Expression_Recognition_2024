import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list

#!!!FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self,
                 alpha = None,
                 gamma = 2,
                 reduction= 'mean'):
        """Constructor.

        Args:
            alpha (float) [C]: class weights. Defaults to 1.
            gamma (float): constant (the higher, the more important are hard examples). Defaults to 2.
            reduction (str): 'mean', 'sum'. Defaults to 'mean'.
        """
        super().__init__()
        self.alpha = alpha if alpha is not None else 1.0
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):  
        """
        Args:
            logits (_type_): [batch_size, num_classes]
            labels (_type_): [batch_size]

        Returns:
            reduced loss 
        """
        log_p = F.log_softmax(logits, dim=1) #?log_softmax for numerical stability
        p = torch.exp(log_p)
 
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        #?select correct class probabilities and alpha, by multiplying for labels_one_hot
        p_t = (p * labels_one_hot).sum(dim=1)
        log_p_t = (log_p * labels_one_hot).sum(dim=1)
        alpha_t = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(self.alpha).to(logits.device)
        alpha_t = (self.alpha * labels_one_hot).sum(dim=1)

        loss = - alpha_t * ((1 - p_t)**self.gamma) * torch.log(log_p_t)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

#!CENTER LOSS
# class CenterLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, device='cpu'):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.device = device
        
#         # Initialize the centers
#         self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device, non_blocking=True))
    
#     def forward(self, features, labels):
#         # Get the centers corresponding to the labels
#         batch_size = features.size(0)
#         centers_batch = self.centers.index_select(0, labels)
        
#         # Calculate the center loss
#         center_loss = F.mse_loss(features, centers_batch)
#         return center_loss
    
# class CEL_CL_Loss(nn.Module):
#     def __init__(self, num_classes, feat_dim, lambda_center=0.5, device='cpu'):
#         super(CEL_CL_Loss, self).__init__()
#         self.cross_entropy_loss = nn.CrossEntropyLoss()
#         self.center_loss = CenterLoss(num_classes, feat_dim, device)
#         self.lambda_center = lambda_center

#     def forward(self, logits, features, labels):
#         ce_loss = self.cross_entropy_loss(logits, labels)
#         c_loss = self.center_loss(features, labels)
#         total_loss = ce_loss + self.lambda_center * c_loss
#         return total_loss
    
    
    
    
