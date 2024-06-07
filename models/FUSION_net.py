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
            self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(512)

            # Fully connected layers
            self.fc1 = nn.Linear(512 * 14 * 14 + 512 * 2, 1024) 
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes)   
            
        
        #!Resnet50
        if args.models['RGB'].model == 'RGB_ResNet50' and args.models['DEPTH'].model == 'DEPTH_ResNet50':
            self.conv1 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(2048)

            # Fully connected layers
            self.fc1 = nn.Linear(2048 * 14 * 14 + 2048 *2, 1024)  
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, num_classes) 

    def forward(self, data):
        rgb_output, rgb_feat  = self.rgb_model(data['RGB'])
        depth_output, depth_feat = self.depth_model(data['DEPTH'])
        
        #resnet18
            #mid: [batch_size, 256, 14, 14] #late: #[batch_size, 512, 1, 1]
        #resnet50
            #mid: [batch_size, 1024, 14, 14] #late: #[batch_size, 2048, 1, 1]
                    
        # Extract mid-level and late-level features
        mid_feat_rgb = rgb_feat['mid_feat']
        mid_feat_depth = depth_feat['mid_feat']
        late_feat_rgb = rgb_feat['late_feat']
        late_feat_depth = depth_feat['late_feat']

        # Concatenate mid-level features
        mid_combined = torch.cat((mid_feat_rgb, mid_feat_depth), dim=1)

        # Apply additional convolutions on mid-level features
        x = F.relu(self.bn1(self.conv1(mid_combined)))
        if hasattr(self, 'conv2'):
            x = F.relu(self.bn2(self.conv2(x)))

        # Flatten mid-level features
        x = torch.flatten(x, 1)

        # Concatenate flattened mid-level features with late-level features
        late_combined = torch.cat((late_feat_rgb, late_feat_depth), dim=1)
        # Flatten late-level features
        late_flat = torch.flatten(late_combined, 1)
        
        combined_features = torch.cat((x, late_flat), dim=1)

        # Apply fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, {}
    
    
    
class Attention(nn.Module):
    def __init__(self, feat_dim):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(feat_dim * 2, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 1)

    def forward(self, rgb_feats, depth_feats):
        combined_feats = torch.cat((rgb_feats, depth_feats), dim=1)
        attention_weights = torch.sigmoid(self.fc2(F.relu(self.fc1(combined_feats))))
        attended_feats = attention_weights * rgb_feats + (1 - attention_weights) * depth_feats
        return attended_feats
    
class Attention_Fusion_net(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(Attention_Fusion_CNN, self).__init__()
        
        #?define RGB and Depth networks (from configuration file)
        self.rgb_model = getattr(model_list, args.models['RGB'].model)()
        self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 
        
        #!Resnet18
        if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':            
            self.attention_mid = Attention(256)  # Adjust based on the feature map size
            self.attention_late = Attention(512)  # Adjust based on the feature map size
            self.fc1 = nn.Linear(512 * 2, 256)
            self.fc2 = nn.Linear(256, num_classes)


        #!Resnet50
        if args.models['RGB'].model == 'RGB_ResNet50' and args.models['DEPTH'].model == 'DEPTH_ResNet50':
            self.attention_mid = Attention(1024)  # Adjust based on the feature map size
            self.attention_late = Attention(2048)  # Adjust based on the feature map size
            self.fc1 = nn.Linear(2048 * 2, 256)
            self.fc2 = nn.Linear(256, num_classes)


    def forward(self, data):
        rgb_output, rgb_feat  = self.rgb_model(data['RGB'])
        depth_output, depth_feat = self.depth_model(data['DEPTH'])
        
        #resnet18
            #mid: [batch_size, 256, 14, 14] #late: #[batch_size, 512, 1, 1]
        #resnet50
            #mid: [batch_size, 1024, 14, 14] #late: #[batch_size, 2048, 1, 1]
        
        # Apply attention to mid-level features
        mid_feats = self.attention_mid(rgb_feat['mid_feat'], depth_feat['mid_feat'])
        
        # Apply attention to late-level feats
        late_feats = self.attention_late(rgb_feat['late_feat'], depth_feat['late_feat'])
        
        # Combine mid and late feats
        combined_feats = torch.cat((mid_feats, late_feats), dim=1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(combined_feats))
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
