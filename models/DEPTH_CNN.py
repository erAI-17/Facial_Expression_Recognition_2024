import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

#!PRETRAINED RESNET-18
class DEPTH_ResNet18(nn.Module):
    def __init__(self):
        super(DEPTH_ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT) 
        
        # Modify the first convolutional layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize the new conv1 layer with pretrained weights
        pretrained_weights = models.resnet18(weights=ResNet18_Weights.DEFAULT).conv1.weight
        self.model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        
        #? Freeze all layers except the conv1, last two residual blocks, and the last fully connected layer
        for name, param in self.model.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x) #?batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #?layers 1,2,3,4 are residual blocks of the ResNet model, each consisting of multiple convolutional layers and skip connections. 
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        mid_feat = self.model.layer3(x) #[batch_size, 256, 14, 14]
        late_feat = self.model.layer4(mid_feat)
        late_feat = self.model.avgpool(late_feat) #[batch_size, 512, 1, 1]
        
        return x, {'mid_feat': mid_feat, 'late_feat': late_feat}
    

#!PRETRAINED RESNET50
class DEPTH_ResNet50(nn.Module):
    def __init__(self):
        super(DEPTH_ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
               
        # Modify the first convolutional layer to accept grayscale images (1 channel instead of 3)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize the new conv1 layer with pretrained weights
        pretrained_weights = models.resnet50(weights=ResNet50_Weights.DEFAULT).conv1.weight
        self.model.conv1.weight.data = pretrained_weights.mean(dim=1, keepdim=True)
        
        #? Freeze all layers except the last two residual blocks and the last fully connected layer
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):        
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #?layers 1,2,3,4 are residual blocks of the ResNet model, each consisting of multiple convolutional layers and skip connections. 
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        mid_feat = self.model.layer3(x)  #[batch_size, 1024, 1, 1]
        late_feat = self.model.layer4(mid_feat)
        late_feat = self.model.avgpool(late_feat) #[batch_size, 2048, 1, 1] 
        
        return x, {'mid_feat': mid_feat, 'late_feat': late_feat}
    