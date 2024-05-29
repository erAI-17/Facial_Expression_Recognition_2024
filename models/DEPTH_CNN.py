import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from torchvision import models


#!PRETRAINED RESNET-18
class DEPTH_CNN(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(DEPTH_CNN, self).__init__()
        self.model = models.resnet18(weights=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Freeze all layers except the conv1, last two residual blocks, and the last fully connected layer
        for name, param in self.model.named_parameters():
            if 'conv1' in name or 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x, {}

# #!RAW
# class DEPTH_CNN(nn.Module):
#     def __init__(self):
#         num_classes, valid_labels= utils.utils.get_domains_and_labels(args)
#         super(DEPTH_CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(128 * 28 * 28, 128)  # 28x28 is the size after pooling
#         self.fc2 = nn.Linear(128, num_classes)  

#     def forward(self, x):   #x.size = tensor[32, 3, 224,224]
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x))) #x.size = tensor[32, 128, 28, 28]
#         feat = x
#         x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
    
#         return x, {}