import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T

class SIMPLER_AttentionFusion1D_Module(nn.Module):
   def __init__(self, C, d_ff):
      super(SIMPLER_AttentionFusion1D_Module, self).__init__()
      self.attention = nn.Linear(C * 2, 1)
      self.ffn = nn.Sequential(
         nn.Linear(C, d_ff),
         nn.ReLU(),
         nn.Linear(d_ff, C)
      )
      #?LayerNorm normalizes across the features for each individual sample.
      #?BatchNorm normalizes across the batch for each feature.
      self.layer_norm = nn.LayerNorm(C)
      self.d_ff = d_ff

   def forward(self, rgb_feat, depth_feat):
      # Concatenate the projected features
      combined_features = torch.cat((rgb_feat, depth_feat), dim=-1)

      # Compute attention weights
      attn_weights = F.softmax(self.attention(combined_features), dim=-1)

      # Apply attention weights
      weighted_rgb = attn_weights * rgb_feat
      weighted_depth = (1 - attn_weights) * depth_feat

      # Fuse features
      fused_features = weighted_rgb + weighted_depth
      fused_features = self.layer_norm(fused_features)

      # Apply feed-forward network
      output = self.ffn(fused_features)
      return output
   

class AttentionFusion1D_Module(nn.Module):
   def __init__(self, C, nhead, d_ff):
      super(AttentionFusion1D_Module, self).__init__()
      self.multihead_attn = nn.MultiheadAttention(embed_dim=C, num_heads=nhead,  batch_first=True)
      self.ffn = nn.Sequential(
         nn.Linear(C, d_ff),
         nn.ReLU(),
         nn.Linear(d_ff, C)
      )
      self.layer_norm = nn.LayerNorm(C)
      self.d_ff = d_ff

   def forward(self, rgb_feat, depth_feat):
      # Concatenate the features
      combined_features = torch.cat((rgb_feat.unsqueeze(1), depth_feat.unsqueeze(1)), dim=1)
      
      #! Apply multi-head attention
      attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features) #expects input: #? [batch_size, sequence_length= H*W+1, dimension=Cp]
      
      attn_output = attn_output.permute(1, 0, 2)  # Back to original shape
      
      attn_output = self.layer_norm(attn_output)

      # Aggregate attention output (e.g., take mean over the sequence)
      fused_features = attn_output.mean(dim=1) 

      # Apply feed-forward network
      output = self.ffn(fused_features)
      return output
    

class AttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(AttentionFusion1D, self).__init__()
      #?define RGB and Depth networks (from configuration file)
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #!EfficientNetB0
      if args.models['RGB'].model == 'efficientnet_b0' and args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
      #!EfficientNetB3
      elif args.models['RGB'].model == 'efficientnet_b3' and args.models['DEPTH'].model == 'efficientnet_b3':
         self.C = 1536
      #!ViT
      elif args.models['RGB'].model == 'ViT':  
         if args.models['DEPTH'].model == 'efficientnet_b0':
            self.C = 1280
         elif args.models['DEPTH'].model == 'efficientnet_b3':
            self.C = 1536
         self.bn = nn.BatchNorm1d(196)
         self.dropout = nn.Dropout(0.2)
         self.project_rgb = nn.Linear(196*768, self.C)
        
      self.attention = SIMPLER_AttentionFusion1D_Module(self.C, nhead=4, d_ff=1024)
      #self.attention = AttentionFusion1D_Module(self.C, d_model=512, nhead=4, d_ff=1024) 
      
      #?final classifier
      self.fc = nn.Linear(512, num_classes) 

   def forward(self, rgb_input, depth_input):
      rgb_feat  = self.rgb_model(rgb_input)['late_feat']
      depth_feat = self.depth_model(depth_input)['late_feat']
      
      #project to common dimension
      if self.project_rgb is not None:
         rgb_feat = self.bn(rgb_feat)
         batch_size, seq_length, input_dim = rgb_feat.shape
         rgb_feat = rgb_feat.view(batch_size,-1) #? flatten
         rgb_feat = self.dropout(rgb_feat)
         rgb_feat = self.project_rgb(rgb_feat)

      # Apply attention
      att_fused_feat = self.attention(rgb_feat, depth_feat)  

      x = self.fc(att_fused_feat)

      return x, {}
