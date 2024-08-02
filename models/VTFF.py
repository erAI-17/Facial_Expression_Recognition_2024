import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
from transformers import BertModel


class AttentionSelectiveFusion_Module(nn.Module):
   """Attention Selective fusion performes fusion by summing the outputs of a GLOBAL fusion and LOCAL fusion.
      LOCAL fusion is a basic attention mechanism with the addition of learnable weight matrices W_L and W_C that initially fuse the 2 modalities.  
      While GLOBAL fusion is the same thing but it receives an average pooled input feature
   """
   def __init__(self, C, reduction_ratio=8):
      super(AttentionSelectiveFusion_Module, self).__init__()
      
      self.C = C
      self.reduction_ratio = reduction_ratio   
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
         
      #? W_RGB  and  W_D  are defined as 1x1 convolutional layers. 
      #? They act as learnable weight matrices that can adaptively adjust the importance of features during fusion. 
      self.W_RGB = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
      self.W_D = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)  
      
      #? GLOBAL context layers
      self.conv1G = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.conv2G = nn.Conv2d(self.C // reduction_ratio, self.C, kernel_size=1, padding=0)
      self.bnG1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.bnG2 = nn.BatchNorm2d(self.C)
      
      #? LOCAL context layers
      self.conv1L = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.conv2L = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bnL1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.bnL2 = nn.BatchNorm2d(1)
 
   def forward(self, rgb_feat, depth_feat):
      
      U = self.W_RGB(rgb_feat) + self.W_D(depth_feat) # [batch_size, C, H, W]
      
      # Global context working on C
      G = self.sigmoid(self.bnG2(self.conv2G(self.relu(self.bnG1(self.conv1G(F.adaptive_avg_pool2d(U, (1,1)))))))) #?[batch_size, C, 1, 1]
      
      # Local context
      L = self.sigmoid(self.bnL2(self.conv2L(self.relu(self.bnL1(self.conv1L(U)))))) #?[batch_size, 1, H, W]
      
      # Combine global and local contexts
      GL = G + L #?[batch_size, C,H,W]
      
      # Fused feature map
      X_fused = rgb_feat * GL + depth_feat * (1 - GL)  #?[batch_size, C, H, W]
      
      return X_fused
   
   

class VTFF(nn.Module):
   """Visual Transformer Feature Fusion with Attention Selective fusion.
      Combines multiple Transformer encoder layers for classification
   """
   def __init__(self, rgb_model, depth_model, p_dropout):
      super(VTFF, self).__init__()      
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      #!EfficientNetB0
      if args.models['RGB'].model == 'efficientnet_b0' and args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
         self.HW = 7
      #!EfficientNetB3
      elif args.models['RGB'].model == 'efficientnet_b3' or args.models['DEPTH'].model == 'efficientnet_b3':
         self.C = 1536
         self.HW = 7
      
      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      self.p_dropout = p_dropout
      
      #? attentional selective fusion module producing X_fused [batch_size, Cf=256 x Hd=14 x Wd=14]
      self.AttentionSelectiveFusion_Module = AttentionSelectiveFusion_Module(self.C)
      
      #? Parameters for the transformer encoder   
      self.Cp = 768
      self.nhead  = 8
      self.num_layers = 4
      self.mlp_dim = 3072 #?MLP dimension INTERNAL to each transformer layer (std is 2048)
      self.seq_len = self.HW**2
      
      self.linear_proj = nn.Linear(self.C,  self.Cp)
      
      self.trans_encoder_layer = nn.TransformerEncoderLayer(self.Cp, nhead=self.nhead, dim_feedforward=self.mlp_dim, activation='gelu', batch_first=True)
      self.tranformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=self.num_layers)
      
      self.cls_token = nn.Parameter(torch.zeros(1, 1,  self.Cp))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1,  self.Cp))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      #? final classification
      self.fc = nn.Linear(self.Cp, num_classes)

   def forward(self, rgb_input, depth_input):
      
      X_rgb  = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input) 
      
      #efficientnet_b0: #[batch_size, 1280, 7, 7]
      #efficientnet_b2: #[batch_size, 1408, 7, 7]
      
      #?X_fused from attentional selective fusion module:  
      X_fused = self.AttentionSelectiveFusion_Module(X_rgb['mid_feat'], X_depth['mid_feat']) #? [batch_size, C x H x W]

      #? Flat and Project linealry to Cp channels [batch_size, C x H*W] ->  [batch_size, Cp x H*W]
      X_fused = X_fused.view(X_fused.size(0), self.C, -1)  # Flat
      X_fused = X_fused.permute(0, 2, 1) #?[batch_size, H*W, Cf] permute beacause nn.Linear reduces tha last dimension (not the channel) and we want to reduce the channels instead 
      X_fused = self.linear_proj(X_fused) #Project #? (batch_size, H*W, Cp)

      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(X_fused.size(0), -1, -1) # (batch_size, 1, Cp)
      X_fused = torch.cat((cls_tokens, X_fused), dim=1) # (batch_size, H*W+1, Cp)

      #?add positional embedding as learnable parameters to each element of the sequence
      X_fused = X_fused + self.pos_embed #(batch_size, H*W+1, Cp)

      X_fused = self.tranformer_encoder(X_fused) #transformer (with batch_first=True) expects input: #? [batch_size, sequence_length= H*W+1, dimension=Cp]
      
      #?classification
      cls_output = X_fused[:, 0]  #?Extract [cls] token's output
      logits = self.fc(cls_output)
      
      return logits, {}


   