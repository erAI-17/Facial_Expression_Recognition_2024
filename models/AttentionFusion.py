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

      #?final classifier
      self.fc = nn.Linear(self.C , num_classes) 

   def forward(self, rgb_input, depth_input):
      X_rgb  = self.rgb_model(rgb_input)['late']       
      X_depth = self.depth_model(depth_input)['late'] #? [batch_size, self.C]
            
      X_fused = torch.add(X_rgb, X_depth)  #? [batch_size, self.C]  
           
      # Classification
      logits = self.fc(X_fused)  

      return logits, {'late': X_fused}

   


