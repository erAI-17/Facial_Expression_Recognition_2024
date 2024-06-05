import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50

class feature_FUSION_net(nn.Module):
    '''
    Naive network that concatenates the features from 2 modalities (RGB and Depth map).
    Still the CONCATENATION + level_averaging fusion is NOT efficient
    
    This handles both Resnet18 and Resnet50
    '''
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(feature_FUSION_net, self).__init__()
        
        #?define RGB and Depth networks (from configuration file)
        self.rgb_model = getattr(model_list, args.models['RGB'].model)()
        self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 

        #!Resnet18
        if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':
            self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(1024)
            self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(512)

            # Fully connected layers
            self.fc1 = nn.Linear(512 * 7 * 7, 256) 
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)  
        
        #!Resnet50
        if args.models['RGB'].model == 'RGB_ResNet50' and args.models['DEPTH'].model == 'DEPTH_ResNet50':
            self.conv1 = nn.Conv2d(4096, 2048, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(2048)
            self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(1024)

            # Fully connected layers
            self.fc1 = nn.Linear(1024 * 7 * 7, 512)  
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes) 

    def forward(self, data):
        rgb_output, rgb_feat  = self.rgb_model(data['RGB']) #?late feat [batch_size:32, 512]
        depth_output, depth_feat = self.depth_model(data['DEPTH'])  #?late feat [batch_size:32, 512]
        
        #concatenate the features at different levels (mid, late) from the RGB and DEEP networks
        combined_features = []
        for level in rgb_feat.keys():
            combined = torch.cat((rgb_feat[level], depth_feat[level]), dim=1)  # Concatenate features
            combined_features.append(combined)

        # Average all levels features
        avg_combined = torch.mean(torch.stack(combined_features), dim=0)

        # Apply additional convolutional layers
        x = F.relu(self.bn1(self.conv1(avg_combined)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the features to prepare them for FC
        x = torch.flatten(x, 1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, {}
    
    
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 2, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 1)

    def forward(self, rgb_features, depth_features):
        combined_features = torch.cat((rgb_features, depth_features), dim=1)
        attention_weights = torch.sigmoid(self.fc2(F.relu(self.fc1(combined_features))))
        attended_features = attention_weights * rgb_features + (1 - attention_weights) * depth_features
        return attended_features
    
class Attention_Fusion_CNN(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(Attention_Fusion_CNN, self).__init__()
        self.RGB_CNN = DEPTH_ResNet50()
        self.DEPTH_CNN = DEPTH_ResNet50()
        self.attention = Attention(512)  # Adjust based on the feature map size
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Assuming 10 classes for classification

    def forward(self, image, d_map):
        rgb_output, rgb_feat = self.RGB_CNN(image)
        depth_output, depth_feat = self.DEPTH_CNN(d_map)
        rgb_features_flat = rgb_feat.view(-1, 512)
        depth_features_flat = depth_feat.view(-1, 512)
        attended_features = self.attention(rgb_features_flat, depth_features_flat)
        x = F.relu(self.fc1(attended_features))
        x = self.fc2(x)
        return x, {}
    
    
#!!!FOCAL LOSS
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        '''
         Args:
            alpha (float): Weighting factor for the rare class. Default is 1.
            gamma (float): Focusing parameter. Default is 2.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy loss
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # The probability of the true class
        pt = torch.exp(-BCE_loss)
        # Focal loss calculation
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
