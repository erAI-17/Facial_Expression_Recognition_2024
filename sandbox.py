import torch
from torch import nn
import torch.nn.functional as F

'''
This file allows for training-test split of dataset and data visualization . 
For training-test split different proportion can be chosen (80-20, 90-10)
For data visualization, it shows 1 sample as:
- 2d image, 
- depth map, 
- generate point cloud from 2d+depth_map, 
- generate triangular mesh from 2d+depth map)
'''


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha = 1,
                 gamma= 2,
                 reduction= 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        p_t = F.softmax(logits, dim=-1)  #softmax, then logarithm
        
        num_classes = logits.size(1)
        labels = F.one_hot(labels, num_classes=num_classes).float()
        
        check0 = (p_t * labels)
        check1 = (p_t * labels).sum(dim=1)
        check2 = torch.log(p_t) * labels
        ce = (torch.log(p_t) * labels * self.alpha).sum(dim=1)

        #focal term: (1 - pt)^gamma
        focal_term = (1 - (p_t * labels).sum(dim=1))**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = -focal_term * ce
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
    
 
#!##
#!MAIN
#!##
if __name__ == '__main__':
    logits = torch.randn(4, 5)
    labels = torch.arange(0, 4)
    
    FocalLoss =  FocalLoss(alpha=1, gamma=2, reduction='sum')
    
    FocalLoss.forward(logits, labels)
    
   
    
        
    
    
    
    
    
    
