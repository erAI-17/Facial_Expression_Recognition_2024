import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args

class GlobalAttention_Module(nn.Module):
   def __init__(self, C, reduction_ratio=8):
      super(GlobalAttention_Module, self).__init__()
      
      self.C = C
      self.reduction_ratio = reduction_ratio   
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
      
      #? GLOBAL context layers
      self.conv1G = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.conv2G = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bnG1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.bnG2 = nn.BatchNorm2d(1)
 
   def forward(self, X_fused):
      
      # Global context
      G = self.sigmoid(self.bnG2(self.conv2G(self.relu(self.bnG1(self.conv1G(F.adaptive_avg_pool2d(X_fused, (1,1)))))))) #?[batch_size, C, 1, 1]
      
      return G


class LocalAttention_Module(nn.Module):
   def __init__(self, C, reduction_ratio=8):
      super(LocalAttention_Module, self).__init__()
      
      self.C = C
      self.reduction_ratio = reduction_ratio   
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
         
      #? LOCAL context layers
      self.conv1 = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.bn1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.conv2 = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1)
 
   def forward(self, X_fused):
      # Local context
      L = self.sigmoid(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(X_fused)))))) #?[batch_size, 1, H, W]
         
      return L
   

class LocalGlobalFusionNet(nn.Module):
   """Visual Transformer Feature Fusion with Attention Selective fusion.
      Combines multiple Transformer encoder layers for classification
   """
   def __init__(self, rgb_model, depth_model):
      super(LocalGlobalFusionNet, self).__init__()      
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
       
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #? Heights and Widths of the feature maps at different stages 
      #self.stages = {'late': [352, 7]} try only late
      self.stages = {'early': [16, 112], 'mid': [88, 14], 'late': [352, 7]} #? mid 88 if efficientnet_b2 else 80
      
      self.num_patches = 7 #?num_patches on 1 dimension, tot num patches: 7 x 7
      self.patch_size = {}
      self.W_RGB = nn.ModuleDict()
      self.W_D = nn.ModuleDict()
      self.LocalAttention_Module = nn.ModuleDict()
      self.GlobalAttention_Module = nn.ModuleDict()
      self.proj_layers = nn.ModuleDict()
      
      for stage in self.stages.keys():
            self.W_RGB[stage] = nn.Conv2d(self.stages[stage][0], self.stages[stage][0], kernel_size=1, bias=False)
            self.W_D[stage] = nn.Conv2d(self.stages[stage][0], self.stages[stage][0], kernel_size=1, bias=False)
            #self.LocalAttention_Module[stage] = LocalAttention_Module(self.stages[stage][0])
            #self.GlobalAttention_Module[stage] = GlobalAttention_Module(self.stages[stage][0])
            
            self.patch_size[stage] = (self.stages[stage][1] // self.num_patches)**2 * self.stages[stage][0]
            self.proj_layers[stage] = nn.Linear(self.patch_size[stage], 768)
               
      #? Transformer encoder   
      self.Ct = 768
      self.nhead  = 8
      self.num_layers = 4
      self.mlp_dim = 3072 #?MLP dimension INTERNAL to each transformer layer (std is 2048)
      self.seq_len = (self.num_patches ** 2) * len(self.stages.keys()) #concatenate early, mid and late features
      
      self.trans_encoder_layer = nn.TransformerEncoderLayer(self.Ct, nhead=self.nhead, dim_feedforward=self.mlp_dim, activation='gelu', batch_first=True)
      self.tranformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=self.num_layers)
      
      self.cls_token = nn.Parameter(torch.zeros(1, 1,  self.Ct))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1,  self.Ct))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      #? final classification
      self.fc = nn.Linear(self.Ct, num_classes)

   def forward(self, rgb_input, depth_input):
      
      X_rgb  = self.rgb_model(rgb_input)      
      X_depth = self.depth_model(depth_input)
      X_fused = {}            
      
      for stage in self.stages.keys():
         X_fused[stage] = self.W_RGB[stage](X_rgb[stage]) + self.W_D[stage](X_depth[stage])
         #X_fused[stage] = X_fused[stage] + X_fused[stage] * self.LocalAttention_Module[stage](X_fused[stage])
         #X_fused[stage] = X_fused[stage] * self.GlobalAttention_Module[stage](X_fused[stage])
        
         #? Extract patches and flatten into sequence like input for transformer
         patch_size = self.stages[stage][1] // 7
         patches = X_fused[stage].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) #? [batch_size, C, H, W] -> [batch_size, C, H//patch_size, W//patch_size, patch_size, patch_size]
         patches = patches.contiguous().view(X_fused[stage].size(0), X_fused[stage].size(1), -1, patch_size, patch_size)  #? [batch_size, C, H*W (num_patches), patch_size, patch_size]
         X_fused[stage] = patches.view(patches.size(0), -1, patch_size * patch_size * X_fused[stage].size(1)) #? [batch_size, H*W, C*patch_size*patch_size]
         
         #? Project features to transformer dimension
         X_fused[stage] = self.proj_layers[stage](X_fused[stage]) #? apply projection layer
         
      #? concatenate early, mid and late features
      X_fused = torch.cat((X_fused['early'], X_fused['mid'], X_fused['late']), dim=1)
      
      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(X_fused.size(0), -1, -1) # (batch_size, 1, C)
      X_fused = torch.cat((cls_tokens, X_fused), dim=1) # (batch_size, H*W+1, C)
      #?add positional embedding as learnable parameters to each element of the sequence
      X_fused = X_fused + self.pos_embed #(batch_size, H*W+1, C)

      X_fused = self.tranformer_encoder(X_fused) #transformer (with batch_first=True) expects input: #? [batch_size, sequence_length= H*W+1, dimension=C]
      
      #?classification
      cls_output = X_fused[:, 0]  #?Extract [cls] token's output
      logits = self.fc(cls_output)

      
      return logits, {'late':cls_output} 

