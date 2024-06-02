import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

#!PRETRAINED RESNET-18
class RGB_ResNet18(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(RGB_ResNet18, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        #? Freeze all layers except the last two residual blocks and the last fully connected layer
        for name, param in self.model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        #x = self.model(x) #?train all network together
        
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        #?layers 1,2,3,4 are residual blocks of the ResNet model, each consisting of multiple convolutional layers and skip connections. 
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        #? Extract features before the last fully connected layer
        late_feat = torch.flatten(x, 1)
        x = self.model.fc(late_feat)
        return x, {'late_feat': late_feat}
    
#! RESNET50
class RGB_ResNet50(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(RGB_ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        #? Freeze all layers except the last two residual blocks and the last fully connected layer
        for name, param in self.model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        #x = self.model(x) #?train all network together
        
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        #?layers 1,2,3,4 are residual blocks of the ResNet model, each consisting of multiple convolutional layers and skip connections. 
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        #? Extract features before the last fully connected layer
        late_feat = torch.flatten(x, 1)
        x = self.model.fc(late_feat)
        return x, {'late_feat': late_feat}
    


