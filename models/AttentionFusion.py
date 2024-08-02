import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T


class ConcatenationFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model, p_dropout):
      super(ConcatenationFusion1D, self).__init__()
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0
      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
      #!EfficientNetB2
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408

      #?final classifier
      self.fc = nn.Linear(self.C , num_classes) 

   def forward(self, rgb_input, depth_input):
      X_rgb  = self.rgb_model(rgb_input)['late_feat']       
      X_depth = self.depth_model(depth_input)['late_feat'] #? [batch_size, self.C]
            
      X_fused = torch.add(X_rgb, X_depth)  #? [batch_size, self.C]  
           
      # Classification
      logits = self.fc(X_fused)  # [batch_size, num_classes]

      return logits, {'late_feat': X_fused}

   
class SequenceAttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model, p_dropout):
      super(SequenceAttentionFusion1D, self).__init__()
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      self.C = 768
      self.nhead  = 8
      self.num_layers = 4
      self.mlp_dim = 2048 #?MLP dimension INTERNAL to each transformer layer (std is 2048)
      self.seq_len = 196
      
      # Transformer encoder layer
      encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True, dim_feedforward=2048, activation='gelu')
      self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
      
      self.cls_token = nn.Parameter(torch.zeros(1, 1,  self.C))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1,  self.C))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      #?final classifier
      self.fc = nn.Linear(768, num_classes) 

   def forward(self, rgb_input, depth_input):
      
      X_rgb  = self.rgb_model(rgb_input)
      X_rgb_cls, X_rgb_feat = X_rgb[0], X_rgb[1]['late_feat'] #? cls: [batch_size, 768], late_feat: [batch_size, 196, 768]
      
      X_depth  = self.depth_model(depth_input)
      X_depth_cls, X_depth_feat = X_depth[0], X_depth[1]['late_feat'] #? cls: [batch_size, 768], late_feat: [batch_size, 196, 768]
      
      # Sum RGB and Depth features
      X_fused = torch.add(X_rgb_feat, X_depth_feat) #?[batch_size, 196, 768]
      
      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(X_fused.size(0), -1, -1) # (batch_size, 1, C)
      X_fused = torch.cat((cls_tokens, X_fused), dim=1) # (batch_size, H*W+1, C)
      #?add positional embedding as learnable parameters to each element of the sequence
      X_fused = X_fused + self.pos_embed #(batch_size, H*W+1, Cp)
      
      X_fused = self.transformer_encoder(X_fused)  #? [batch_size, 768]
            
      #? classification
      X_fused_cls = X_fused[:, 0]  #?Extract [cls] token's output
      logits = self.fc(X_fused_cls)

      return logits, {'late_feat': X_fused}  


