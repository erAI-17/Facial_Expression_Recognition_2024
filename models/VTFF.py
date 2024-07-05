import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50


class AttentionSelectiveFusion_Module(nn.Module):
   """Attention Selective fusion performes fusion by summing the outputs of a GLOBAL fusion and LOCAL fusion.
      LOCAL fusion is a basic attention mechanism with the addition of learnable weight matrices W_L and W_C that initially fuse the 2 modalities.  
      While GLOBAL fusion is the same thing but it receives an average pooled input feature
   """
   def __init__(self, rgb_model, depth_model, reduction_ratio=8):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(AttentionSelectiveFusion_Module, self).__init__()
      self.reduction_ratio = reduction_ratio
      
      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #!Resnet18
      if args.models['RGB'].model == 'RGB_ResNet18' or args.models['DEPTH'].model == 'DEPTH_ResNet18':    
         feat_dim = 512        
      #!Resnet50
      elif args.models['RGB'].model == 'RGB_ResNet50' or args.models['DEPTH'].model == 'DEPTH_ResNet50':
         feat_dim = 2048    
         
      #? W_RGB  and  W_D  are defined as 1x1 convolutional layers. 
      #? They act as learnable weight matrices that can adaptively adjust the importance of features during fusion. 
      self.W_RGB = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
      self.W_D = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)  
      
      #? GLOBAL context layers
      self.conv1G = nn.Conv2d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, padding=0)
      self.conv2G = nn.Conv2d(feat_dim // reduction_ratio, feat_dim, kernel_size=1, padding=0)
      self.bnG1 = nn.BatchNorm2d(feat_dim // reduction_ratio)
      self.bnG2 = nn.BatchNorm2d(feat_dim)
      
      #? LOCAL context layers
      self.conv1L = nn.Conv2d(feat_dim, feat_dim // reduction_ratio, kernel_size=1, padding=0)
      self.conv2L = nn.Conv2d(feat_dim // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bnL1 = nn.BatchNorm2d(feat_dim // reduction_ratio)
      self.bnL2 = nn.BatchNorm2d(1)

      #?classification (for ablation study: not using transformer)
      self.fc1 = nn.Linear(feat_dim*7*7, 1024)
      self.fc2 = nn.Linear(1024, 512)
      self.fc3 = nn.Linear(512, num_classes)
      
      
   def forward(self, data):
      _, rgb_feat  = self.rgb_model(data['RGB'])
      _, depth_feat = self.depth_model(data['DEPTH'])
      
      #resnet18: #[batch_size, 512, 7, 7] 
      #resnet50: #[batch_size, 2048, 7, 7]
      
      U = self.W_RGB(rgb_feat['feat']) + self.W_D(depth_feat['feat']) # [batch_size, 256, 14, 14]
      
      # Global context working on channels
      G = nn.Sigmoid(self.bnG2(self.conv2G(nn.ReLU(self.bnG1(self.conv1G(F.adaptive_avg_pool2d(U, (1,1)))))))) #[batch_size, C, 1, 1]
      
      # Local context
      L = nn.Sigmoid(self.bnL2(self.conv2L(nn.ReLU(self.bnL1(self.conv1L(U)))))) #[batch_size, 1, H, W]
      
      # Combine global and local contexts
      GL = G + L #[batch_size, C,H,W]
      
      # Fused feature map
      X_fused = rgb_feat['feat'] * GL + depth_feat['feat'] * (1 - GL)  #?[batch_size, C=2048 x H=7 x W=7]
      
      #?classification (for ablation study: not using transformer)
         #?Flatten the input feature matrix
      x = X_fused.view(X_fused.size(0), -1) #batch_size, -1
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)

      return x, {'X_fused': X_fused}

class VTFF(nn.Module):
   """Visual Transformer Feature Fusion with Attention Selective fusion.
      Combines multiple Transformer encoder layers for classification
   """

   Cp = 768
   nhead  = 8
   num_layers = 4
   mlp_dim = 3072 #?MLP dimension INTERNAL to each transformer layer
   def __init__(self, Cp, nhead , num_layers, mlp_dim):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(VTFF, self).__init__()
      #? attentional selective fusion module producing X_fused [batch_size, Cf=256 x Hd=14 x Wd=14]
      self.AttentionSelectiveFusion_Module = AttentionSelectiveFusion_Module()
      
      #!Resnet18
      if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':
         self.Cf = 512 #?channel dimension from resnet
      #!Resnet50
      elif args.models['RGB'].model == 'RGB_ResNet50' or args.models['DEPTH'].model == 'DEPTH_ResNet50':
         self.Cf = 2048 #?channel dimension from resnet
      
      self.h = 7 #?height and width dimension of feature (equal for Resnet18 and Resnet50)
      self.seq_len = self.h**2
      
      self.cls_token = nn.Parameter(torch.zeros(1, 1, Cp))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, Cp))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      
      self.linear_proj = nn.Linear(self.Cf, Cp)
      self.trans_encoder_layer = nn.TransformerEncoderLayer(Cp, nhead=nhead, dim_feedforward=mlp_dim, activation='gelu', batch_first=True)
      self.tranformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=num_layers)
      
      #? final classification
      self.droput = nn.Dropout(0.5)
      self.fc = nn.Linear(Cp, num_classes)

   def forward(self, x):
      #?extract X_fused from attentional selective fusion module: X_fused [batch_size, Cf=2048 x H=7 x W=7]
      _, X_fused = self.AttentionSelectiveFusion_Module(x)
      X_fused = X_fused['X_fused']
      
      #? Flat and Project linealry to Cp channels [batch_size, Cf=2048 x H*W=7*7] ->  [batch_size, Cp=768 x H*W=7*7]
      x = x.view(X_fused.size(0), self.Cf, -1)  # Flat
      x = x.permute(0, 2, 1) # (batch_size, H*W, Cp)
      x = self.linear_proj(x) # Linear projection

      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # (batch_size, 1, Cp)
      x = torch.cat((cls_tokens, x), dim=1) # (batch_size, seq_len+1, Cp)

      #?add positional embedding as learnable parameters to each element of the sequence
      x = x + self.pos_embed

      x = self.tranformer_encoder(x) 
      
      #?classification
      cls_output = x[:, 0]  #?Extract [cls] token's output
      x = nn.ReLU(cls_output)
      x = self.dropout(x)
      logits = self.fc(cls_output)
      
      return logits, {}


   