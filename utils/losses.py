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
                 alpha = 1,
                 gamma = 2,
                 reduction= 'sum'):
        """Constructor.

        Args:
            alpha (float) [C]: class weights. Defaults to 1.
            gamma (float): constant (the higher, the more important are hard examples). Defaults to 2.
            reduction (str): 'mean', 'sum'. Defaults to 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        p_t = F.softmax(logits, dim=-1)  
        
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes)
        
        p_t = (self.alpha * p_t * labels_one_hot).sum(dim=1)
        ce = -torch.log(p_t)

        focal_term = (1 - p_t)**self.gamma

        loss = focal_term * ce #?-alpha * ((1 - pt)^gamma) * log(pt)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
#
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
    
    
    
    
