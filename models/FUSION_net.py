import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T

from models.RGB_CNN import RGB_CNN
from models.DEPTH_CNN import DEPTH_CNN


class feature_level_concat_FUSION_net(nn.Module):
    '''
    Naive network that concatenates the features from 2 modalities (RGB and Depth map).
    Performs better than logit level fusion, but requires additional training.
    Still the CONCATENATION fusion is NOT efficient
    '''
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(feature_level_concat_FUSION_net, self).__init__()
        self.rgb_model = RGB_CNN()  # Define the RGB network
        self.depth_model = DEPTH_CNN()  # Define the Depth network
        self.fc1 = nn.Linear(512 * 2, 128)  # Adjust based on the feature map size
        self.fc2 = nn.Linear(128, num_classes)  # Assuming 10 classes for classification

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
        
        x = F.relu(self.fc1(avg_combined))
        x = self.fc2(x)
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
        self.RGB_CNN = RGB_CNN()
        self.DEPTH_CNN = DEPTH_CNN()
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