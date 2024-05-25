import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T

from models.RGB_CNN import RGB_CNN
from models.DEPTH_CNN import DEPTH_CNN

class logit_level_concat_FUSION_net(nn.Module):
    '''
    Naive network that concatenates the logits from 2 modalities (RGB and Depth map).
    Simpler, Faster but not very performing
    '''
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(logit_level_concat_FUSION_net, self).__init__()
        self.rgb_model = RGB_CNN()  # Define the RGB network
        self.depth_model = DEPTH_CNN()  # Define the Depth network
        self.fc1 = nn.Linear(num_classes + num_classes, 128)  # Fully connected layer after concatenation
        self.fc2 = nn.Linear(128, num_classes)  # Final fully connected layer for classification

    def forward(self, image, d_map):
        rgb_output, rgb_feat = self.rgb_model(image)  # Get logits from RGB network
        depth_output, depth_feat = self.depth_model(d_map)  # Get logits from Depth network
        combined = torch.cat((rgb_output, depth_output), dim=1)  # Concatenate logits
        x = F.relu(self.fc1(combined))  # Pass through first fully connected layer with ReLU activation
        x = self.fc2(x)  # Pass through final fully connected layer for classification
        return x
    
    
class feature_level_concat_FUSION_net(nn.Module):
    '''
    Naive network that concatenates the logits from 2 modalities (RGB and Depth map).
    Performs better than logit level fusion, but requires additional training.
    Still the CONCATENATION fusion is NOT efficient
    '''
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(logit_level_concat_FUSION_net, self).__init__()
        self.rgb_model = RGB_CNN()  # Define the RGB network
        self.depth_model = DEPTH_CNN()  # Define the Depth network
        self.fc1 = nn.Linear(128 * 28 * 28 * 2, 128)  # Adjust based on the feature map size
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification

    def forward(self, image, d_map):
        rgb_output, rgb_feat  = self.rgb_model(image) 
        depth_output, depth_feat = self.depth_model(d_map)  
        combined = torch.cat((rgb_feat, depth_feat), dim=1)  # Concatenate features
        combined = combined.view(-1, 128 * 28 * 28 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        return x
    
    
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
        super(Attention_Fusion_CNN, self).__init__()
        self.rgb_model = RGB_CNN()
        self.depth_model = DEPTH_CNN()
        self.attention = Attention(128 * 28 * 28)  # Adjust based on the feature map size
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification

    def forward(self, image, d_map):
        rgb_features = self.rgb_model(image)
        depth_features = self.depth_model(d_map)
        rgb_features_flat = rgb_features.view(-1, 128 * 28 * 28)
        depth_features_flat = depth_features.view(-1, 128 * 28 * 28)
        attended_features = self.attention(rgb_features_flat, depth_features_flat)
        x = F.relu(self.fc1(attended_features))
        x = self.fc2(x)
        return x