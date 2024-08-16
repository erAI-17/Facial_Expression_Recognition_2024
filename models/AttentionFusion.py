import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T


class SumFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(SumFusion1D, self).__init__()
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0
      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
      #!EfficientNetB2
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408
      #!MobilNetv4
      elif args.models['DEPTH'].model == 'mobilenet_v4':
         self.C = 1280
      
      #?final classifier
      self.fc = nn.Linear(self.C , num_classes) 

   def forward(self, rgb_input, depth_input):
      X_rgb, _  = self.rgb_model(rgb_input)     
      X_depth, _ = self.depth_model(depth_input) #? [batch_size, self.C]
            
      X_fused = torch.add(X_rgb, X_depth)  #? [batch_size, self.C]  
           
      # Classification
      logits = self.fc(X_fused)  

      return logits, {'late': X_fused}


class AttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(AttentionFusion1D, self).__init__()
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model, _ = rgb_model
      self.depth_model, _ = depth_model

      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408

      self.Att_map_rgb = nn.Linear(self.C, self.C)
      self.Att_map_depth = nn.Linear(self.C, self.C)
      
      self.fc1 = nn.Linear(self.C*2 , self.C)
      self.bn1 = nn.BatchNorm1d(self.C)
      self.dropout = nn.Dropout(p=args.models['FUSION'].dropout)
      self.fc2 = nn.Linear(self.C , num_classes)
      
   def forward(self, rgb_input, depth_input):
      X_rgb  = self.rgb_model(rgb_input)       
      X_depth = self.depth_model(depth_input)
      
      Att_X_rgb = torch.sigmoid(self.Att_map_rgb(X_rgb))
      Att_X_depth = torch.sigmoid(self.Att_map_depth(X_depth))
      
      X_rgb = Att_X_rgb * X_rgb
      X_depth = Att_X_depth * X_depth
      
      X_fused = torch.cat((X_rgb, X_depth), dim=1)
      X_fused = self.dropout(F.relu(self.bn1(self.fc1(X_fused))))
      logits = self.fc2(X_fused)
      
      return logits, {'late': X_fused}
   
   
class CrossAttentionFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(CrossAttentionFusion1D, self).__init__()
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      if args.models['DEPTH'].model == 'efficientnet_b0':
         self.C = 1280
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408

      self.cross_attention_rgb = nn.MultiheadAttention(embed_dim=self.C, num_heads=8)
      self.cross_attention_depth = nn.MultiheadAttention(embed_dim=self.C, num_heads=8)
      
      self.fc1 = nn.Linear(self.C * 2, self.C)
      self.bn1 = nn.BatchNorm1d(self.C)
      self.dropout = nn.Dropout(p=0.5)
      self.fc2 = nn.Linear(self.C, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)

      # Cross-attention
      #   # Reshape for MultiheadAttention: (L, N, E) where L is sequence length, N is batch size, E is embedding dimension. Returns the output and the attention weights
      Att_X_rgb, _ = self.cross_attention_rgb(X_rgb.unsqueeze(0), X_depth.unsqueeze(0), X_depth.unsqueeze(0))
      Att_X_depth, _ = self.cross_attention_depth(X_depth.unsqueeze(0), X_rgb.unsqueeze(0), X_rgb.unsqueeze(0))

      # Remove sequence length dimension
      Att_X_rgb = Att_X_rgb.squeeze(0)
      Att_X_depth = Att_X_depth.squeeze(0)

      # Concatenate features
      X_fused = torch.cat((Att_X_rgb, Att_X_depth), dim=1)

      # Fully connected layers
      X_fused = self.dropout(F.relu(self.bn1(self.fc1(X_fused))))
      logits = self.fc2(X_fused)

      return logits, {'late': X_fused}
