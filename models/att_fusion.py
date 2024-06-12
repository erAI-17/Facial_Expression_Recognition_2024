import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50

class basic_attention(nn.Module):
   """#? fc1 is FC layer (or 1x1 convolution) that reduces the number of channels from 2C to a smaller intermediate dimension.
      #? fc2 is another FC layer (or 1x1 convolution) that brings the number of channels back to C.
      #? sigmoid ensures the attention weights are in the range [0, 1].
   By concatenating the features and applying fully connected layers, the network learns a set of attention weights that can dynamically focus on different feature maps.
   Example:
   input: [batch_size x C x H x W] 
   output: [batch_size x 1 x H x W] because 1x1 convolutions (don't change H and W) and  
   """
   def __init__(self, feat_dim, reduction_ratio=8):
      super(basic_attention, self).__init__()
      self.fc1 = nn.Conv2d(feat_dim * 2, feat_dim//reduction_ratio, kernel_size=1, padding=0) #? kernel_size=1 doesn't modify input HxW !
      self.bn1 = nn.BatchNorm2d(feat_dim // reduction_ratio) 
      self.fc2 = nn.Conv2d(feat_dim//reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1) 

   def forward(self, rgb_feats, depth_feats):
      #?concatenate along channel dimension
      combined_feats = torch.cat((rgb_feats, depth_feats), dim=1)  #?[batch_size, C, H, W] -> [batch_size, 2*C, H, W]
      
      #? fc1 is FC layer (or 1x1 convolution) that reduces the number of channels from 2C to a smaller intermediate dimension defined by reduction ration.
      #? fc2 is another FC layer (or 1x1 convolution) that brings the number of channels to 1, so we have a matrix 1xHxW of attention weights.
      #? sigmoid ensures the attention weights are in the range [0, 1].
      attention_weights = torch.sigmoid(self.bn2(self.fc2(F.relu(self.bn1(self.fc1(combined_feats)))))) #?[batch_size, 2*C, H, W] -> fc1:[batch_size, C/r, H, W] -> fc2:[batch_size, 1, H, W]
      
      attended_feats = attention_weights * rgb_feats + (1 - attention_weights) * depth_feats
      
      return attended_feats
    
class basic_att_fusion(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(basic_att_fusion, self).__init__()
        
        #?define RGB and Depth networks (from configuration file)
        self.rgb_model = getattr(model_list, args.models['RGB'].model)()
        self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 
        
        #!Resnet18
        if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':            
            self.attention_mid = basic_attention(256, reduction_ratio=8) 
            self.attention_late = basic_attention(512, reduction_ratio=8)  
            self.fc1 = nn.Conv2d(512 * 2, 256)
            self.fc2 = nn.Conv2d(256, num_classes)

        #!Resnet50
        if args.models['RGB'].model == 'RGB_ResNet50' and args.models['DEPTH'].model == 'DEPTH_ResNet50':
            self.attention_mid = basic_attention(1024, reduction_ratio=8) 
            self.attention_late = basic_attention(2048, reduction_ratio=8) 
            self.fc1 = nn.Conv2d(2048 * 2, 256)
            self.fc2 = nn.Conv2d(256, num_classes)

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
    


class att_sel_fusion(nn.Module):
   """Attention Selective fusion performes fusion by summing the outputs of a GLOBAL fusion and LOCAL fusion.
      LOCAL fusion is exactly the same thing implemented in " basic_attention " module with the only addition of learnable weight matrices W_L and W_C that initially fuse the 2 modalities.
      instead in "basic_attention" I just concatenate the features from 2 modalities at beginning
      
      While GLOBAL fusion is more or less the same thing but it receives ana verage pooled input feature
      
      Initially I try only with mid features extracted after layer 3 of resnet18
      
      #resnet18
         #mid: [batch_size, 256, 14, 14] #late: #[batch_size, 512, 1, 1]
      #resnet50
         #mid: [batch_size, 1024, 14, 14] #late: #[batch_size, 2048, 1, 1]
   """
   def __init__(self, feat_dim, reduction_ratio=8):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(att_sel_fusion, self).__init__()
      self.reduction_ratio = reduction_ratio
      self.feat_dim = feat_dim 
      
      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = getattr(model_list, args.models['RGB'].model)()
      self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 
      
      #? GLOBAL context layers
      self.conv1G = nn.Conv2d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, padding=0)
      self.conv2G = nn.Conv2d(feat_dim // reduction_ratio, feat_dim, kernel_size=1, padding=0)
      self.bnG1 = nn.BatchNorm2d(feat_dim // reduction_ratio)
      self.bnG2 = nn.BatchNorm2d(feat_dim)
      
      #? LOCAL context layers
      self.conv1L = nn.Conv2d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, padding=0)
      self.conv2L = nn.Conv2d(feat_dim // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bnL1 = nn.BatchNorm2d(feat_dim // reduction_ratio)
      self.bnL2 = nn.BatchNorm2d(1)
      
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU()

      self.fc1 = nn.Conv2d(512, 128)
      self.fc2 = nn.Conv2d(128, num_classes)
      
   def forward(self, data):
      rgb_output, rgb_feat  = self.rgb_model(data['RGB'])
      depth_output, depth_feat = self.depth_model(data['DEPTH'])
      
      #? W_L  and  W_C  are defined as 1x1 convolutional layers. 
      #? They act as learnable weight matrices that can adaptively adjust the importance of features during fusion. 
      W_RGB = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False).cuda()
      W_D = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1, bias=False).cuda()
      
      U = W_RGB(rgb_feat['mid_feat']) + W_D(depth_feat['mid_feat'])
      
      # Global context
      G = self.sigmoid(self.bnG2(self.conv2G(self.relu(self.bnG1(self.conv1G(F.adaptive_avg_pool2d(U, 1)))))))
      
      # Local context
      L = self.sigmoid(self.bnL2(self.conv2L(self.relu(self.bnL1(self.conv1L(U))))))
      
      # Combine global and local contexts
      GL = G + L
      
      # Fused feature map
      X_fused = rgb_feat['mid_feat'] * GL + depth_feat['mid_feat'] * (1 - GL)  #?[batch_size, Cf=512 x Hd x Wd]
      
      #?classification (for ablation study: not using transformer)
      x = F.relu(self.fc1(X_fused))
      x = self.fc2(x)
      
      return x, {'fused_feat': X_fused}
   