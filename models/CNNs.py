import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from utils.args import args
from torchvision.models import EfficientNet_B0_Weights,EfficientNet_B2_Weights
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
import timm

class efficientnet_b0(nn.Module):
    def __init__(self):
        super(efficientnet_b0, self).__init__()
        
        # Load the model from the .pt file
        self.model = torch.load('./models/pretrained_models/enet_b0_8_best_vgaf.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

        # #? print model
        # with open('enet_b0_8_best_vgaf.txt', 'w') as f:
        #     f.write(str(self.model)) 
        
        self.features  = {}   
        
        # Define hooks to extract features from different layers
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook    

        # Registering hooks to the layers
        self.model.blocks[0].register_forward_hook(get_features('early'))  # Early features
        self.model.blocks[3].register_forward_hook(get_features('mid'))    # Mid features
        self.model.blocks[6].register_forward_hook(get_features('late'))   # Late features

    
    def forward(self, x):       
        _ = self.model(x)  # Forward pass through the model to trigger hooks
               
        # Extracted features are already stored in self.features
        X_early = self.features.get('early') #? [batch_size, 16, 112, 112]
        X_mid = self.features.get('mid') #? [batch_size, 80, 14, 14]
        X_late = self.features.get('late') #? [batch_size, 352, 7, 7]
    
        return {'early': X_early, 'mid': X_mid, 'late': X_late}
    


class efficientnet_b2(nn.Module):
    def __init__(self):
        super(efficientnet_b2, self).__init__()
        
        self.model = torch.load('./models/pretrained_models/enet_b2_7.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

        # #? print model
        # with open('enet_b0_8_best_vgaf.txt', 'w') as f:
        #     f.write(str(self.model)) 
        
        # with open('INDEXES_enet_b2_7.txt', 'w') as f:
        #     for idx, block in enumerate(self.model.blocks):
        #         f.write(f'{idx}: {block}\n')
            
        self.features  = {}   
        
        # Define hooks to extract features from different layers
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook    

        # Registering hooks to the layers
        self.model.blocks[0].register_forward_hook(get_features('early')) 
        self.model.blocks[3].register_forward_hook(get_features('mid')) 
        self.model.blocks[6].register_forward_hook(get_features('late')) 

    
    def forward(self, x):       
        _ = self.model(x)  # Forward pass through the model to trigger hooks
               
        # Extracted features are already stored in self.features
        X_early = self.features.get('early') #? [batch_size, 16, 112, 112]
        X_mid = self.features.get('mid') #? [batch_size, 88, 14, 14]
        X_late = self.features.get('late') #? [batch_size, 352, 7, 7]
    
        return {'early': X_early, 'mid': X_mid, 'late': X_late}
    
    