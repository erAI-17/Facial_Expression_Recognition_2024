import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights,EfficientNet_B3_Weights
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

class efficientnet_b0(nn.Module):
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) 
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2]) #remove [avgpool layer, fc layer]
        self.avgpool =  nn.Sequential(*list(self.model.children())[-2:-1]) #bring [avgpool layer]
        
        # with open('FULL_b0.txt', 'w') as f:
        #     f.write(str(self.model))
        # with open('FEAT_b0.txt', 'w') as f:
        #     f.write(str(self.feature_extractor))

    def forward(self, x):              
        mid_feat = self.feature_extractor(x)
        late_feat = self.avgpool(mid_feat).squeeze()
        
        return x, {'late_feat': late_feat,'mid_feat': mid_feat}
    
    
class efficientnet_b3(nn.Module):
    def __init__(self):
        super(efficientnet_b3, self).__init__()
        
        self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT) 
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2]) #remove [avgpool layer, fc layer]
        self.avgpool =  nn.Sequential(*list(self.model.children())[-2:-1]) #bring [avgpool layer],

    def forward(self, x):
        mid_feat = self.feature_extractor(x)
        late_feat = self.avgpool(mid_feat).squeeze()
        
        return x, {'late_feat': late_feat,'mid_feat': mid_feat}