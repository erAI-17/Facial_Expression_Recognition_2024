import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

#!PRETRAINED RESNET-18
class DEPTH_ResNet18(nn.Module):
    def __init__(self):
        super(DEPTH_ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT) 

    def forward(self, x):
        #stack the image to have 3 channel instead of 1
        x = torch.cat([x, x, x], dim=1)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x) #?batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #?4 stages
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x) 
        feat = self.model.layer4(feat) #[batch_size, 512, 7, 7]
        
        return x, {'feat': x}
    

#!PRETRAINED RESNET50
class DEPTH_ResNet50(nn.Module):
    def __init__(self):
        super(DEPTH_ResNet50, self).__init__()
        
        outer_model = AutoModelForImageClassification.from_pretrained("KhaldiAbderrhmane/resnet50-facial-emotion-recognition", trust_remote_code=True) 
        self.model = outer_model.resnet
        
        #self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  #download pretrained weights? ==True, ResNet18_Weights.DEFAULT TO GET THE MOST UPDATED WEIGHTS
        
        # check the weights
        # with open('IMAGENETresnet50_weights.txt', 'w') as f:
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             f.write(f'Layer Name: {name}\n')
        #             f.write(f'Weights:\n{param.data}\n\n')

    def forward(self, x):        
        #stack the image to have 3 channel instead of 1
        x = torch.cat([x, x, x], dim=1)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x) #batch normalization
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #?4 stages
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x) 
        feat = self.model.layer4(x) #[batch_size, 2048, 7, 7]
        
        return x, {'feat': feat}
    