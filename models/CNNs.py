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
        #? remove fc classifier
        self.model = nn.Sequential(*list(self.model.children())[:-1]) 
        
        # #? print model
        # with open('enet_b0_8_best_vgaf.txt', 'w') as f:
        #     f.write(str(self.model)) 
        
        # Define hooks to extract features from different layers
        self.features  = {}   
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook    

        # Registering hooks to the layers
        self.model[1].register_forward_hook(get_features('early'))  # Early features
        self.model[2][3].register_forward_hook(get_features('mid'))    # Mid features
        self.model[2][6].register_forward_hook(get_features('late'))   # Late features

    
    def forward(self, X):       
        X = self.model(X)  # Forward pass through the model to trigger hooks
        X = X.squeeze() 
              
        # Extracted features are already stored in self.features
        X_early = self.features.get('early') #? [batch_size, 16, 112, 112]
        X_mid = self.features.get('mid') #? [batch_size, 80, 14, 14]
        X_late = self.features.get('late') #? [batch_size, 352, 7, 7]
    
        return X, {'early': X_early, 'mid': X_mid, 'late': X_late}
    


class efficientnet_b2(nn.Module):
    def __init__(self):
        super(efficientnet_b2, self).__init__()
        
        self.model = torch.load('./models/pretrained_models/enet_b2_7.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
        #? remove fc classifier
        self.model = nn.Sequential(*list(self.model.children())[:-1]) 

        # #? print model
        # with open('enet_b0_8_best_vgaf.txt', 'w') as f:
        #     f.write(str(self.model)) 
    
        # Define hooks to extract features from different layers
        self.features  = {}   
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook    

        # Registering hooks to the layers
        self.model[1].register_forward_hook(get_features('early')) 
        self.model[2][3].register_forward_hook(get_features('mid')) 
        self.model[3].register_forward_hook(get_features('late'))  #[2][6] #[3]
        
    def forward(self, X):       
        X = self.model(X)  # Forward pass through the model to trigger hooks
         #squeeze width and height dimensions
       
               
        # Extracted features are already stored in self.features
        X_early = self.features.get('early') #? [batch_size, 32, 130, 130]
        X_mid = self.features.get('mid') #? [batch_size, 88, 17, 17]
        X_late = self.features.get('late') #? [batch_size, 352, 9, 9]
    
        return X, {'early': X_early, 'mid': X_mid, 'late': X_late}
              
    
    
    
class SqueezeExcite_Module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite_Module, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
    
    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class mobilenet_v4(nn.Module):
    def __init__(self):
        super(mobilenet_v4, self).__init__()
        
        self.model = timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True)
        #? remove identity (act2), flatten, fc classifier
        self.model = nn.Sequential(*list(self.model.children())[:-3]) 
        
        #? Replace Identity SE blocks with actual SE blocks
        self._replace_se_blocks()
        
        #? Define hooks to extract features from different layers
        self.features  = {}   
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook    
        # Registering hooks to the layers
        self.model[1].register_forward_hook(get_features('early')) #? [batch_size, 32, 112, 112]
        self.model[2][2].register_forward_hook(get_features('mid')) #?[batch_size, 160, 14, 14]
        self.model[2][4].register_forward_hook(get_features('late')) #? [batch_size, 960, 7, 7]
        
        # #? print model
        # with open('mobilenetv4_conv_medium.txt', 'w') as f:
        #     f.write(str(self.model)) 
       
    def _replace_se_blocks(self):
        previous_out_channels = None
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Identity) and 'se' in module_name:
                parent_name, _ = module_name.rsplit('.', 1)
                parent_module = dict(self.model.named_modules())[parent_name]
                
                # Use the output channels of the previous block
                if previous_out_channels is not None:
                    in_channels = previous_out_channels
                else:
                    raise AttributeError(f"Cannot determine in_channels for module {parent_name}")
                
                se_block = SqueezeExcite_Module(in_channels)
                setattr(parent_module, 'se', se_block)
            
            # Update previous_out_channels if the module is a Conv2d layer
            if isinstance(module, nn.Conv2d):
                previous_out_channels = module.out_channels
    
    def forward(self, X):       
        X = self.model(X)  # Forward pass through the model to trigger hooks
        X = X.squeeze()
               
        # Extracted features are already stored in self.features
        X_early = self.features.get('early') #? [batch_size, 32, 112, 112]
        X_mid = self.features.get('mid') #? [batch_size, 160, 14, 14]
        X_late = self.features.get('late') #? [batch_size, 960, 7, 7]
    
        return X, {'early': X_early, 'mid': X_mid, 'late': X_late}
    