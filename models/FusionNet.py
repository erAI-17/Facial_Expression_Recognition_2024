import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args


class SpatialAttentionModule(nn.Module):
   def __init__(self, C, reduction_ratio=8):
      super(SpatialAttentionModule, self).__init__()
      
      self.C = C
      self.reduction_ratio = reduction_ratio   
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
         
      #? LOCAL context layers
      self.conv1 = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.bn1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.conv2 = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1)
 
   def forward(self, X):
      # Local context
      #L = self.sigmoid(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(X)))))) #?[batch_size, 1, H, W]
      L = self.sigmoid(self.conv2(self.relu(self.conv1(X)))) #?[batch_size, 1, H, W]
  
      return L

class PatchEmbedding(nn.Module):
   def __init__(self, H, patch_size, in_channels, embed_dim=768):
      super(PatchEmbedding, self).__init__()
      self.H = H
      self.patch_size = patch_size
      self.num_patches = (H // patch_size) ** 2
      self.embed_dim = embed_dim
      
      # Flatten and linear projection for each patch
      self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
   
   def forward(self, x):
      x = self.proj(x)  # (B, embed_dim, H/P, W/P)
      x = x.flatten(2)  # (B, embed_dim, num_patches)
      x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
      return x

class FusionNet(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(FusionNet, self).__init__()      
      num_classes, _ = utils.utils.get_domains_and_labels(args)
       
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #? Heights and Widths of the feature maps at different stages 
      self.stages = {'late': [1408, 9]} # only late features
      #self.stages = {'early': [32, 130], 'mid': [88, 17], 'late': [1408, 9]}
      
      #? Weights for RGB and Depth modalities
      self.weight_rgb = nn.ParameterDict({stage: nn.Parameter(torch.randn(1)) for stage in self.stages})
      self.weight_depth = nn.ParameterDict({stage: nn.Parameter(torch.randn(1)) for stage in self.stages}) 
            
      self.patch_sizes = { 'early': 13 , 'mid': 8, 'late': 1}
      self.n_spatial_attentions = 3
      self.patch_size = {}
      self.Att_map_RGB = nn.ModuleDict()
      self.Att_map_D = nn.ModuleDict()
      
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })
      
      #patches embeddings
      self.patch_embed = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_sizes[stage], self.stages[stage][0]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_sizes[stage], self.stages[stage][0]) for stage in self.stages})
      })

      #? Transformer encoder   
      self.Ct = 768
      self.trans_encoder_layer = nn.TransformerEncoderLayer(self.Ct, nhead=8, dim_feedforward=3072, activation='gelu', batch_first=True)
      self.transformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=8)
      self.cls_token = nn.Parameter(torch.zeros(1, 1,  self.Ct))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1,  sum([(self.stages[stage][1] // self.patch_sizes[stage])**2 for stage in self.stages]) + 1,  self.Ct))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      #? final classification
      self.fc = nn.Linear(self.Ct, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)
      
      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}  # Assume X_rgb[1] and X_depth[1] contain 'early', 'mid', 'late' stages

      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            
            #? apply multiple spatial attention modules and perform maxpooling
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attetion_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attetion_module(X[m][stage])), dim=1)
            
            #max pooling over all channels
            X_att = torch.max(X_att, dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
            #batch norm and Parametric relu
            X_att = F.relu(nn.BatchNorm2d(X_att.size(1)).to(X_att.device)(X_att))
            
            X[m][stage] = X_att * X[m][stage] #?apply attention to features
         
            #?patch embeddings
            X[m][stage] = self.patch_embed[m][stage](X[m][stage])
         
      #?concatenate the modalitites   
      X_fused_global = torch.cat([X['rgb'][stage]*self.weight_rgb[stage] + X['depth'][stage]*self.weight_depth[stage] for stage in self.stages], dim=1)
      
      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(X_fused_global.size(0), -1, -1) # (batch_size, 1, C)
      X_fused_global = torch.cat((cls_tokens, X_fused_global), dim=1) # (batch_size, H*W+1, C)
      #?add positional embedding as learnable parameters to each element of the sequence
      X_fused_global = X_fused_global + self.pos_embed #(batch_size, H*W+1, C)

      X_fused_global = self.transformer_encoder(X_fused_global) #transformer (with batch_first=True) expects input: #? [batch_size, sequence_length= H*W+1, dimension=C]
      
      #?classification
      cls_output = X_fused_global[:, 0]  #?Extract [cls] token's output
      logits = self.fc(cls_output)

      return logits, {'late': cls_output} 

