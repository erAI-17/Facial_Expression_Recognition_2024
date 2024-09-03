import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification



class SpatialAttentionModule(nn.Module):
   def __init__(self, C, reduction_ratio=8):
      super(SpatialAttentionModule, self).__init__()
      
      self.C = C
      self.reduction_ratio = reduction_ratio   
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()
         
      self.conv1 = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.bn1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.conv2 = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1)
 
   def forward(self, X):
      L = self.sigmoid(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(X)))))) #?[batch_size, 1, H, W]
      #L = self.sigmoid(self.conv2(self.relu(self.conv1(X)))) #?[batch_size, 1, H, W]
  
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

      num_classes = utils.utils.get_domains_and_labels(args)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      self.pretrained_vit = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

      # Extract specific layers of the ViT encoder
      self.vit_layers = nn.ModuleList(self.pretrained_vit.encoder.layer[:-4])

      # Update the number of tokens and dimensions based on stages
      self.patch_size = {'early': 13, 'mid': 8, 'late': 1}
      #self.stages = {'late': [352, 9]}  # Adjust this according to the stages you want to use
      self.stages = {'early': [32, 130], 'mid': [88, 17], 'late': [352, 9]}
      
      self.n_spatial_attentions = 5
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })

      #?patches embeddings
      self.patch_embed = nn.ModuleDict({
      'rgb': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_size[stage], self.stages[stage][0]) for stage in self.stages}),
      'depth': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_size[stage], self.stages[stage][0]) for stage in self.stages})
      })

      # Final classification head
      self.fc = nn.Linear(768, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)

      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}
      
      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            
            #? apply multiple spatial attention modules and perform maxpooling
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attention_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attention_module(X[m][stage])), dim=1)
               
            #? max pooling over all channels   
            X_att = torch.max(X_att, dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
            X[m][stage] = X_att * X[m][stage]
            
            #?patch embeddings
            X[m][stage] = self.patch_embed[m][stage](X[m][stage])

      #? Concatenate features from both modalities and stages
      X_fused = torch.zeros(rgb_input.size(0), 0, 768).to(X['rgb']['late'].device) 
      for stage in self.stages:
        X_fused = torch.cat((X_fused, (X['rgb'][stage] + X['depth'][stage])), dim=1)

      #? Pass through selected ViT layers
      for layer in self.vit_layers:
        X_fused = layer(X_fused)[0]  # [batch_size, num_patches, Ct]

      #? Classification using the [CLS] token
      cls_output = X_fused[:, 0]  # [CLS] token output
      logits = self.fc(cls_output)

      return logits, {'late': cls_output}
   
   
   
class FusionNetCross(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(FusionNetCross, self).__init__()

      num_classes = utils.utils.get_domains_and_labels(args)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      self.pretrained_vit = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

      # Extract specific layers of the ViT encoder
      self.vit_layers = nn.ModuleList(self.pretrained_vit.encoder.layer[:-4])

      # Update the number of tokens and dimensions based on stages
      self.patch_size = {'early': 13, 'mid': 8, 'late': 1}
      self.stages = {'early': [32, 130], 'mid': [88, 17], 'late': [352, 9]}
      
      self.n_spatial_attentions = 5
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleList([SpatialAttentionModule(self.stages['late'][0]) for _ in range(self.n_spatial_attentions)]),
         'depth': nn.ModuleList([SpatialAttentionModule(self.stages['late'][0]) for _ in range(self.n_spatial_attentions)])
      })

      #?patches embeddings
      self.patch_embed = PatchEmbedding(self.stages['late'][1], self.patch_size['late'], self.stages['late'][0])

      # Final classification head
      self.fc = nn.Linear(768, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)

      X = {'rgb': X_rgb[1]['late'], 'depth': X_depth[1]['late']}
      
      X_att = {m: torch.zeros(X[m].size(0), 0, X[m].size(2), X[m].size(3)).to(X[m].device) for m in ['rgb', 'depth']}
      for m in ['rgb', 'depth']:          
         #? apply multiple spatial attention modules and perform maxpooling
         for spatial_attention_module in self.SpatialAttentionModules[m]:
            X_att[m] = torch.cat((X_att[m], spatial_attention_module(X[m])), dim=1)
            
         #? max pooling over all channels   
         X_att[m] = torch.max(X_att[m], dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
      #average the attention maps
      X_att_avg = (X_att['rgb'] + X_att['depth']) / 2
      
      X_fused = X_att_avg * X['rgb']
         
      #?patch embeddings
      X_fused = self.patch_embed(X_fused)

      #? Pass through selected ViT layers
      for layer in self.vit_layers:
        X_fused = layer(X_fused)[0]  # [batch_size, num_patches, Ct]

      #? Classification using the [CLS] token
      cls_output = X_fused[:, 0]  # [CLS] token output
      logits = self.fc(cls_output)

      return logits, {'late': cls_output}