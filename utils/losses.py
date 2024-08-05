import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list

class CE_Center_Criterion(nn.Module):
    def __init__(self, ce_loss=None, center_loss=None, lambda_center=0.1):
        super(CE_Center_Criterion, self).__init__()
        self.ce_loss = ce_loss
        self.center_loss = center_loss
        self.lambda_center = lambda_center

    def forward(self, logits, labels, features):
        ce_loss_value = self.ce_loss(logits, labels)
        center_loss_value = self.center_loss(features, labels)
        total_loss = ce_loss_value + self.lambda_center/2 * center_loss_value
        return total_loss

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

        loss = - alpha_t * ((1 - p_t)**self.gamma) * log_p_t
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none': 
            pass
        return loss

#!CENTER LOSS
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.num_classes, _ = utils.utils.get_domains_and_labels(args)
        self.use_gpu = True if torch.cuda.is_available() else False
        self.centers = None

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size, feat_dim = x.size()
        
        # Initialize centers if not already done
        if self.centers is None or self.centers.size(1) != feat_dim:
            if self.use_gpu:
                self.centers = nn.Parameter(torch.randn(self.num_classes, feat_dim).cuda())
            else:
                self.centers = nn.Parameter(torch.randn(self.num_classes, feat_dim))
        
        # Compute the distance between features and centers
        #?||x - c||^2 = ||x||^2 + ||c||^2 - (2 * x * c)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t()) #? add and mult (-2 * x * c)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).mean() #? sum over batch_size and compute mean

        return loss
    
    
