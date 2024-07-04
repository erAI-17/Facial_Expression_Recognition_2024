import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list

#!!!FOCAL LOSS
class FocalLoss1(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        """    
        Args:
            logits (torch.Tensor): probabilistic logits with shape (batch_size, num_classes)
            labels (torch.Tensor): classes (batch_size)
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
        """
        pred_softmax = F.softmax(logits, dim=1)
        
        num_classes = logits.size(1)
        labels = F.one_hot(labels, num_classes=num_classes + 1)
        labels = labels[:, :num_classes]
        labels = labels.type_as(logits) #?cast to same type as logits
        
        pt = (1 - pred_softmax) * labels + pred_softmax * (1 - labels)
        focal_weight = (self.alpha * labels + (1 - self.alpha) *
                        (1 - labels)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred_softmax, labels, reduction='none') * focal_weight
        
        #reduce loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Initialize the centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, features, labels):
        # Get the centers corresponding to the labels
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        
        # Calculate the center loss
        center_loss = F.mse_loss(features, centers_batch)
        return center_loss
    
class CEL_CL_Loss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_center=0.5, device='cpu'):
        super(CEL_CL_Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(num_classes, feat_dim, device)
        self.lambda_center = lambda_center

    def forward(self, logits, features, labels):
        ce_loss = self.cross_entropy_loss(logits, labels)
        c_loss = self.center_loss(features, labels)
        total_loss = ce_loss + self.lambda_center * c_loss
        return total_loss
    
    
    
    
class FocalLoss2(nn.Module):
    def __init__(self,
                 alpha = None,
                 gamma= 0.,
                 reduction= 'mean'):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none')

    def forward(self, x, y):
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        #all_rows = torch.arange(len(x))
        log_pt = log_p[:, y] 

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss