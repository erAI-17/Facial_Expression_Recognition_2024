import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
   
class SelfAttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model, p_dropout):
      super(SelfAttentionFusion1D, self).__init__()
      #?define RGB and Depth networks (from configuration file)
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0
      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.depth_dim = 1280
      #!EfficientNetB3
      elif args.models['DEPTH'].model == 'efficientnet_b3':
         self.depth_dim = 1536

      # Project EfficientNet features to the common dimension
      self.effnet_proj = nn.Linear(self.depth_dim, 768)

      # Transformer encoder layer
      encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True, dim_feedforward=2048, activation='gelu')
      self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
      
      #?final classifier
      self.fc = nn.Linear(768, num_classes) 

   def forward(self, rgb_input, depth_input):
      rgb_vit_output  = self.rgb_model(rgb_input)
      cls, rgb_feat = rgb_vit_output[0], rgb_vit_output[1]['late_feat'] #? cls: [batch_size, 768], late_feat: [batch_size, 196, 768]
      
      depth_feat = self.depth_model(depth_input)['late_feat'] #? [batch_size, C]
      
      depth_feat = self.effnet_proj(depth_feat)  #? [batch_size, 768]
      depth_feat = depth_feat.unsqueeze(1)  #? [batch_size, 1, 768]
      
      # Concatenate ViT and EfficientNet features
      x = torch.cat((rgb_feat, depth_feat), dim=1)  #? [batch_size, 197, 768]
      #x = cls + depth_feat  #? [batch_size, 197, 768]
        
      # Apply transformer encoder
      x = self.transformer_encoder(x)  #? [batch_size, 197, 768]
      
      # Global average pooling
      x = x.mean(dim=1)  #?[batch_size, 768]
      
      # Classification
      x = self.fc(x)  # [batch_size, num_classes]

      return x, {}



class ConcatenationFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model, p_dropout):
      super(ConcatenationFusion1D, self).__init__()
      #?define RGB and Depth networks (from configuration file)
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0
      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.depth_dim = 1280
      #!EfficientNetB3
      elif args.models['DEPTH'].model == 'efficientnet_b3':
         self.depth_dim = 1536

      # Project EfficientNet features to the common dimension
      self.effnet_proj = nn.Linear(self.depth_dim, 768)

      # Transformer encoder layer
      encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True, dim_feedforward=2048, activation='gelu')
      self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
      
      #?final classifier
      self.fc = nn.Linear(768, num_classes) 

   def forward(self, rgb_input, depth_input):
      rgb_vit_output  = self.rgb_model(rgb_input)
      cls, rgb_feat = rgb_vit_output[0], rgb_vit_output[1]['late_feat'] #? cls: [batch_size, 768], late_feat: [batch_size, 196, 768]
      
      depth_feat = self.depth_model(depth_input)['late_feat'] #? [batch_size, C]
      
      depth_feat = self.effnet_proj(depth_feat)  #? [batch_size, 768]
      depth_feat = depth_feat.unsqueeze(1)  #? [batch_size, 1, 768]
      
      # Concatenate ViT and EfficientNet features
      x = torch.cat((rgb_feat, depth_feat), dim=1)  #? [batch_size, 197, 768]
      #x = cls + depth_feat  
        
      # Apply transformer encoder
      #x = self.transformer_encoder(x)  #? [batch_size, 197, 768]
      
      # Global average pooling
      x = x.mean(dim=1)  #?[batch_size, 768]
      
      # Classification
      x = self.fc(x)  # [batch_size, num_classes]

      return x, {}
