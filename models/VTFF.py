import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
import models as model_list
from models.RGB_CNN import RGB_ResNet18, RGB_ResNet50
from models.DEPTH_CNN import DEPTH_ResNet18, DEPTH_ResNet50


class TransformerEncoderLayer(nn.Module):
   """SINGLE Transformer encoder layer, consisting of multi-head self-attention and a MLP.
   """
   def __init__(self, Cp, num_heads, mlp_dim, dropout=0.1):
      super(TransformerEncoderLayer, self).__init__()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
      self.norm1 = nn.LayerNorm(Cp)
      self.norm2 = nn.LayerNorm(Cp)
      
   def forward(self, x):
      #?MultiheadAttention self-attention method requires three inputs: the query, key, and value matrices.
      #? In self-attention, these matrices are the same, so (x,x,x)
      x2 = self.self_attn(x, x, x)[0] #? returns attention scores nad attention weights, so select only outputs [0]
      
      x = x + self.dropout(x2) #?residual connection
      
      x = self.norm1(x) #?normalization
      #? first Linear layer expands the dimensionality of the features to a higher-dimensional space (mlp_dim). 
      #? This helps in capturing more complex patterns and interactions within the data.
      #? first Linear layer projects the features back to the original dimensionality (Cp). 
      #? This transformation allows the model to combine the learned features and interactions back into the original feature space.
      x2 = self.linear2(self.dropout(F.gelu(self.linear1(x))))
      
      x = x + self.dropout(x2) #?residual connection
      
      x = self.norm2(x)
      
      return x

class VTFF(nn.Module):
   """Visual transformer Feature Fusion with Attention Selective fusion.
      Combines multiple Transformer encoder layers for classification
   """

   Cp = 768
   num_heads = 8
   num_layers = 4
   mlp_dim = 3072 #?MLP dimension INTERNAL to each transformer layer
   def __init__(self, Cp, num_heads, num_layers, mlp_dim):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(VTFF, self).__init__()
      #? attentional selective fusion module producing X_fused [batch_size, Cf=256 x Hd=14 x Wd=14]
      self.att_sel_fusion = att_sel_fusion()
      
      #!Resnet18
      if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':
         self.Cf = 256 #?channel dimension from resnet
         self.H = 14 #?height and width dimension from resnet18
         self.seq_len = self.h**2
      
      self.cls_token = nn.Parameter(torch.zeros(1, 1, Cp))
      self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, Cp))
      
      self.input_proj = nn.Linear(self.Cf, Cp)
      self.layers = nn.ModuleList([TransformerEncoderLayer(Cp, num_heads, mlp_dim) for _ in range(num_layers)])
      self.norm = nn.LayerNorm(Cp)
      self.fc = nn.Linear(Cp, num_classes)

   def forward(self, data):
      #?extract X_fused from attentional selective fusion module: X_fused [batch_size, Cf=256 x Hd=14 x Wd=14]
      X_fused = att_sel_fusion(data)
      
      #? Reshape [batch_size, Cf=256 x Hd=14 x Wd=14] ->  [batch_size, Cp=768 x Hd*Wd=14*14] and project input
      x = x.view(X_fused.size(0), -1, self.Cp)  # Reshape to (batch_size, seq_len, Cp)
      x = self.input_proj(x)  # Linear projection

      #?prepend [cls] token
      cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
      x = torch.cat((cls_tokens, x), dim=1)

      #?add positional embedding as learnable parameters to each element of the sequence
      x = x + self.pos_embed

      #?Pass through multi-headed Transformer layers
      for layer in self.layers:
         x = layer(x)

      #?Normalize 
      x = self.norm(x)
      
      #?classification
      cls_output = x[:, 0]  #?Extract [cls] token's output
      logits = self.fc(cls_output)
      
      return logits, {}


class att_sel_fusion(nn.Module):
   """Attention Selective fusion performes fusion by summing the outputs of a GLOBAL fusion and LOCAL fusion.
      LOCAL fusion is exactly the same thing implemented in " basic_attention " module with the only addition of learnable weight matrices W_L and W_C that initially fuse the 2 modalities.
      instead in "basic_attention" I just concatenate the features from 2 modalities at beginning
      
      While GLOBAL fusion is more or less the same thing but it receives ana verage pooled input feature
      
      Initially I try only with mid features extracted after layer 3 of resnet18
   """
   def __init__(self, reduction_ratio=8):
      num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
      super(att_sel_fusion, self).__init__()
      self.reduction_ratio = reduction_ratio
      
      #?define RGB and Depth networks (from configuration file)
      self.rgb_model = getattr(model_list, args.models['RGB'].model)()
      self.depth_model = getattr(model_list, args.models['DEPTH'].model)() 
      
      #!Resnet18
      if args.models['RGB'].model == 'RGB_ResNet18' and args.models['DEPTH'].model == 'DEPTH_ResNet18':    
         feat_dim = 256        
         mid_feat_dim = 256
         late_feat_dim = 512
      #!Resnet50
      elif args.models['RGB'].model == 'RGB_ResNet50' and args.models['DEPTH'].model == 'DEPTH_ResNet50':
         feat_dim = 1024    
         mid_feat_dim = 1024
         late_feat_dim = 2048
         
      #? W_L  and  W_C  are defined as 1x1 convolutional layers. 
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
      
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU()

      #?classification (for ablation study: not using transformer)
      self.fc1 = nn.Linear(256*14*14, 256)
      self.fc2 = nn.Linear(256, num_classes)
      
      
   def forward(self, data):
      rgb_output, rgb_feat  = self.rgb_model(data['RGB'])
      depth_output, depth_feat = self.depth_model(data['DEPTH'])
      
      #resnet18
         #mid: [batch_size, 256, 14, 14] #late: #[batch_size, 512, 1, 1]
      #resnet50
         #mid: [batch_size, 1024, 14, 14] #late: #[batch_size, 2048, 1, 1]
      
      U = self.W_RGB(rgb_feat['mid_feat']) + self.W_D(depth_feat['mid_feat'])
      
      # Global context
      G = self.sigmoid(self.bnG2(self.conv2G(self.relu(self.bnG1(self.conv1G(F.adaptive_avg_pool2d(U, 1))))))) #[batch_sizee, C,1,1]
      
      # Local context
      L = self.sigmoid(self.bnL2(self.conv2L(self.relu(self.bnL1(self.conv1L(U)))))) #[batch_size, 1,H,W]
      
      # Combine global and local contexts
      GL = G + L #[batch_size, C,H,W]
      
      # Fused feature map
      X_fused = rgb_feat['mid_feat'] * GL + depth_feat['mid_feat'] * (1 - GL)  #?[batch_size, Cf=256 x Hd=14 x Wd=14]
      
      #?classification (for ablation study: not using transformer)
         #?Flatten the input feature matrix
      x = X_fused.view(X_fused.size(0), -1) #batch_size, -1
      #x = F.relu(self.fc1(X_fused))
      #x = self.fc2(x)
      
      return x, {'fused_feat': X_fused}
   