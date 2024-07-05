import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification


#!PRETRAINED RESNET-18
class RGB_ResNet18(nn.Module):
    def __init__(self):
        super(RGB_ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #?layers 1,2,3,4 are residual blocks of the ResNet model, each consisting of multiple convolutional layers and skip connections. 
        x = self.model.layer1(x)
        x = self.model.layer2(x)
       #?4 stages
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x) 
        feat = self.model.layer4(feat) #[batch_size, 512, 7, 7]
        
        return x, {'feat': feat}
    
#!PRETRAINED RESNET50
class RGB_ResNet50(nn.Module):
    def __init__(self):
        super(RGB_ResNet50, self).__init__()
        
        outer_model = AutoModelForImageClassification.from_pretrained("KhaldiAbderrhmane/resnet50-facial-emotion-recognition", trust_remote_code=True) 
        self.model = outer_model.resnet
        
        #self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        #?4 stages
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x) 
        feat = self.model.layer4(feat) #[batch_size, 2048, 7, 7]
        
        return x, {'feat': feat}
    


