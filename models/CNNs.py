import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from utils.args import args
from torchvision.models import EfficientNet_B0_Weights,EfficientNet_B2_Weights
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
import timm

class efficientnet_b0(nn.Module):
    def __init__(self, p_dropout):
        super(efficientnet_b0, self).__init__()
        
        self.p_dropout = p_dropout
        
        if args.pretrained:
            # Load the weights from the .pt file
            self.model = torch.load('./models/pretrained_models/enet_b0_8_best_vgaf.pt', map_location='gpu' if torch.cuda.is_available() else 'cpu')
        else:    
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) 
        
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2]) #remove [avgpool layer, fc layer]
        self.avgpool =  nn.Sequential(*list(self.model.children())[-2:-1]) #bring [avgpool layer]
        
        #? print model
        with open('removed_enet_b0_8_best_vgaf.txt', 'w') as f:
            f.write(str(self.model)) 

    def forward(self, x):              
        mid_feat = self.feature_extractor(x)
        late_feat = self.avgpool(mid_feat).squeeze()
    
        return {'late_feat': late_feat, 'mid_feat': late_feat}
    



class efficientnet_b2(nn.Module):
    def __init__(self, p_dropout):
        super(efficientnet_b0, self).__init__()
        
        self.p_dropout = p_dropout
        
        if args.pretrained:
            self.model = models.efficientnet_b2(pretrained=False)
            # Load the weights from the .pt file
            checkpoint = torch.load('pretrained_models/----.pt')
            self.model.load_state_dict(checkpoint)
        else:    
            self.model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT) 
        
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2]) #remove [avgpool layer, fc layer]
        self.avgpool =  nn.Sequential(*list(self.model.children())[-2:-1]) #bring [avgpool layer]
        
        #? print model
        # with open('FULL_b0.txt', 'w') as f:
        #     f.write(str(self.model)) 

    def forward(self, x):              
        mid_feat = self.feature_extractor(x)
        late_feat = self.avgpool(mid_feat).squeeze()
        
        return {'late_feat': late_feat, 'mid_feat': mid_feat}
    
    