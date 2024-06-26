import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50

class attention(nn.Module):
   """#? fc1 is FC layer (or 1x1 convolution) that reduces the number of channels from 2C to a smaller intermediate dimension.
      #? fc2 is another FC layer (or 1x1 convolution) that brings the number of channels back to C.
      #? sigmoid ensures the attention weights are in the range [0, 1].
   By concatenating the features and applying fully connected layers, the network learns a set of attention weights that can dynamically focus on different feature maps.
   Example:
   input: [batch_size x C x H x W] 
   output: [batch_size x 1 x H x W] because 1x1 convolutions (don't change H and W) and  
   """
   def __init__(self, feat_dim, reduction_ratio=8):
      super(attention, self).__init__()
      feat_dim = feat_dim +768
      self.fc1 = nn.Conv2d(feat_dim, feat_dim//reduction_ratio, kernel_size=1, padding=0) #? kernel_size=1 doesn't modify input HxW !
      self.bn1 = nn.BatchNorm2d(feat_dim // reduction_ratio) 
      self.fc2 = nn.Conv2d(feat_dim//reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1) 

   def forward(self, rgb_feats, depth_feats):
      #?concatenate along channel dimension
      combined_feats = torch.cat((rgb_feats, depth_feats), dim=1)  #?[batch_size, C, H, W] -> [batch_size, C', H, W] C'=512+768
      
      #? fc1 is FC layer (or 1x1 convolution) that reduces the number of channels from 2C to a smaller intermediate dimension defined by reduction ration.
      #? fc2 is another FC layer (or 1x1 convolution) that brings the number of channels to 1, so we have a matrix 1xHxW of attention weights.
      #? sigmoid ensures the attention weights are in the range [0, 1].
      attention_weights = torch.sigmoid(self.bn2(self.fc2(F.relu(self.bn1(self.fc1(combined_feats)))))) #?[batch_size, 2*C, H, W] -> fc1:[batch_size, C/reduction_ratio, H, W] -> fc2:[batch_size, 1, H, W]
      
      attended_feats = attention_weights * rgb_feats + (1 - attention_weights) * depth_feats
      
      return attended_feats
    
class att_fusion(nn.Module):
   def __init__(self):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(att_fusion, self).__init__()

      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = getattr(model_list, args.models['RGB'].model)()
      self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 
        
      #!Resnet18
      if args.models['RGB'].model == 'RGB_ResNet18' or args.models['DEPTH'].model == 'DEPTH_ResNet18':            
         mid_feat_dim = 256
         late_feat_dim = 512
      #!Resnet50
      elif args.models['RGB'].model == 'RGB_ResNet50' or args.models['DEPTH'].model == 'DEPTH_ResNet50':
         mid_feat_dim = 1024
         late_feat_dim = 2048
           
      self.attention = attention(late_feat_dim, reduction_ratio=8) 
      self.fc1 = nn.Linear(late_feat_dim * 2, 256)
      self.fc2 = nn.Linear(256, 128)
      self.fc3 = nn.Linear(256, num_classes)

   def forward(self, data):
      rgb_output, rgb_feat  = self.rgb_model(data['RGB'])
      depth_output, depth_feat = self.depth_model(data['DEPTH'])
      
      #resnet18
         #mid: [batch_size, 256, 14, 14] #late: #[batch_size, 512, 1, 1]
      #resnet50
         #mid: [batch_size, 1024, 14, 14] #late: #[batch_size, 2048, 1, 1]
   
      # Apply attention to late-level feats
      attended_feats = self.attention(rgb_feat['late_feat'], depth_feat['late_feat'])
      
      #?squeeze to prepare for fully connected layers
      attended_feats = attended_feats.squeeze()
      
      x = F.relu(self.fc1(attended_feats))
      x = F.relu(self.fc2(attended_feats))
      x = self.fc3(x)
      
      return x, {}
    

