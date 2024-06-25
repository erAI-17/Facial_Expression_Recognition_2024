import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50

class feat_fusion(nn.Module):
    '''
    Naive network that concatenates the features from 2 modalities (RGB and Depth map).
    This handles both Resnet18 and Resnet50
    '''
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(feat_fusion, self).__init__()
        
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
            self.bn = nn.BatchNorm2d(2048)

            # Fully connected layers
            self.fc1 = nn.Linear(2048 * 14 * 14 + 2048 *2,4096) 
            self.fc2 = nn.Linear(4096, 2048) 
            self.fc3 = nn.Linear(2048, 512)
            self.fc4 = nn.Linear(512, num_classes) 

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

        # Concatenate 2 modalities mid-level features
        mid_combined = torch.cat((mid_feat_rgb, mid_feat_depth), dim=1)

        # Apply additional convolutions on mid-level features
        x = F.relu(self.bn(self.conv1(mid_combined)))
        x = F.relu(self.bn(self.conv2(x)))

        # Flatten mid-level features
        x = torch.flatten(x, 1)

        # Concatenate 2 modalities late-level features
        late_combined = torch.cat((late_feat_rgb, late_feat_depth), dim=1)
        # Flatten late-level features
        late_flat = torch.flatten(late_combined, 1)
        
        # Concatenate flattened mid-level features with late-level features
        combined_features = torch.cat((x, late_flat), dim=1)

        # Apply fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x, {}
    
    
    
  