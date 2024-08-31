import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T


class SumFusion1D(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(SumFusion1D, self).__init__()
      num_classes, _ = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0, MobileNetV4
      if args.models['DEPTH'].model == 'efficientnet_b0' or args.models['DEPTH'].model == 'mobilenet_v4':
         self.C = 1280
      #!EfficientNetB2
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408
      
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
      num_classes, _ = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model

      #!EfficientNetB0, MobileNetV4
      if args.models['DEPTH'].model == 'efficientnet_b0' or args.models['DEPTH'].model == 'mobilenet_v4':
         self.C = 1280
      #!EfficientNetB2
      elif args.models['DEPTH'].model == 'efficientnet_b2':
         self.C = 1408

      self.Att_map_rgb = nn.Linear(self.C, self.C)
      self.Att_map_depth = nn.Linear(self.C, self.C)
      
      self.fc1 = nn.Linear(self.C*2 , self.C)
      self.bn1 = nn.BatchNorm1d(self.C)
      self.dropout = nn.Dropout(p=args.models['FUSION'].dropout)
      self.fc2 = nn.Linear(self.C , num_classes)
      
   def forward(self, rgb_input, depth_input):
      X_rgb, _  = self.rgb_model(rgb_input)       
      X_depth, _ = self.depth_model(depth_input)
      
      Att_X_rgb = torch.sigmoid(self.Att_map_rgb(X_rgb))
      Att_X_depth = torch.sigmoid(self.Att_map_depth(X_depth))
      
      X_rgb = Att_X_rgb * X_rgb
      X_depth = Att_X_depth * X_depth
      
      X_fused = torch.cat((X_rgb, X_depth), dim=1)
      
      X_fused = self.dropout(F.relu(self.bn1(self.fc1(X_fused))))
                       
      logits = self.fc2(X_fused)
      
      return logits, {'late': X_fused}
   
