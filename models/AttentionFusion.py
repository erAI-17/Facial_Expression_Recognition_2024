import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list



class SIMPLER_AttentionFusion1D_Module(nn.Module):
   def __init__(self, rgb_dim, depth_dim, d_model, nhead, d_ff):
      super(SIMPLER_AttentionFusion1D_Module, self).__init__()
      self.proj_rgb = nn.Linear(rgb_dim, d_model)
      self.proj_depth = nn.Linear(depth_dim, d_model)
      self.attention = nn.Linear(d_model * 2, 1)
      self.ffn = nn.Sequential(
         nn.Linear(d_model, d_ff),
         nn.ReLU(),
         nn.Linear(d_ff, d_model)
      )
      #?LayerNorm normalizes across the features for each individual sample.
      #?BatchNorm normalizes across the batch for each feature.
      self.layer_norm = nn.LayerNorm(d_model)
      self.d_ff = d_ff

   def forward(self, rgb_feat, depth_feat):
      # Project features to common dimension
      rgb_proj = self.proj_rgb(rgb_feat)
      depth_proj = self.proj_depth(depth_feat)

      # Concatenate the projected features
      combined_features = torch.cat((rgb_proj, depth_proj), dim=-1)

      # Compute attention weights
      attn_weights = F.softmax(self.attention(combined_features), dim=-1)

      # Apply attention weights
      weighted_rgb = attn_weights * rgb_proj
      weighted_depth = (1 - attn_weights) * depth_proj

      # Fuse features
      fused_features = weighted_rgb + weighted_depth
      fused_features = self.layer_norm(fused_features)

      # Apply feed-forward network
      output = self.ffn(fused_features)
      return output
   

class AttentionFusion1D_Module(nn.Module):
   def __init__(self, rgb_dim, depth_dim, d_model, nhead, d_ff):
      super(AttentionFusion1D_Module, self).__init__()
      self.proj_rgb = nn.Linear(rgb_dim, d_model)
      self.proj_depth = nn.Linear(depth_dim, d_model)
      self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
      self.ffn = nn.Sequential(
         nn.Linear(d_model, d_ff),
         nn.ReLU(),
         nn.Linear(d_ff, d_model)
      )
      #?LayerNorm normalizes across the features for each individual sample.
      #?BatchNorm normalizes across the batch for each feature.
      self.layer_norm = nn.LayerNorm(d_model)
      self.d_ff = d_ff

   def forward(self, rgb_feat, depth_feat):
      # Project features to common dimension
      rgb_proj = self.proj_rgb(rgb_feat)
      depth_proj = self.proj_depth(depth_feat)

      # Concatenate the projected features
      combined_features = torch.cat((rgb_proj.unsqueeze(1), depth_proj.unsqueeze(1)), dim=1)

      #! Apply multi-head attention
      # attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
      # attn_output = self.layer_norm(attn_output)

      # Aggregate attention output (e.g., take mean over the sequence)
      fused_features = combined_features.mean(dim=1) #attn_output

      # Apply feed-forward network
      output = self.ffn(fused_features)
      return output
    

class AttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(AttentionFusion1D, self).__init__()
      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      self.attention = SIMPLER_AttentionFusion1D_Module(768, 512, d_model=512, nhead=4, d_ff=1024) # AttentionFusion1D_Module SIMPLER_AttentionFusion1D_Module
      self.fc = nn.Linear(512, num_classes) 

   def forward(self, x):
      _, rgb_feat  = self.rgb_model(x['RGB'])
      _, depth_feat = self.depth_model(x['DEPTH'])
      
      # Apply attention
      att_fused_feat = self.attention(rgb_feat['late_feat'], depth_feat['late_feat'])  

      x = self.fc(att_fused_feat)

      return x, {}
