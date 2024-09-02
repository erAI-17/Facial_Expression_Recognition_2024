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
         
      #? LOCAL context layers
      self.conv1 = nn.Conv2d(self.C, self.C // reduction_ratio, kernel_size=1, padding=0)
      self.bn1 = nn.BatchNorm2d(self.C // reduction_ratio)
      self.conv2 = nn.Conv2d(self.C // reduction_ratio, 1, kernel_size=1, padding=0)
      self.bn2 = nn.BatchNorm2d(1)
 
   def forward(self, X):
      # Local context
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
      
      #? Heights and Widths of the feature maps at different stages 
      self.stages = {'late': [352, 9]} # only late features
      #self.stages = {'early': [32, 130], 'mid': [88, 17], 'late': [352, 9]}
                  
      self.patch_sizes = { 'early': 13 , 'mid': 8, 'late': 1}
      self.n_spatial_attentions = 5
      self.patch_size = {}
        
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })
      
      #?patches embeddings
      self.patch_embed = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_sizes[stage], self.stages[stage][0]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_sizes[stage], self.stages[stage][0]) for stage in self.stages})
      })

      # #? Transformer encoder   
      self.Ct = 768
      self.trans_encoder_layer = nn.TransformerEncoderLayer(self.Ct, nhead=4, dim_feedforward=3072, activation='gelu', batch_first=True)
      self.transformer_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=2)
      self.cls_token = nn.Parameter(torch.zeros(1, 1,  self.Ct))
      nn.init.normal_(self.cls_token, std=0.02)  # Initialize with small random values to break symmetry
      self.pos_embed = nn.Parameter(torch.zeros(1,  sum([(self.stages[stage][1] // self.patch_sizes[stage])**2 for stage in self.stages]) + 1,  self.Ct))
      nn.init.normal_(self.pos_embed, std=0.02)  # Initialize with small random values to break symmetry
      
      #? final classification
      self.fc = nn.Linear(self.Ct, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)
      
      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}  # Assume X_rgb[1] and X_depth[1] contain 'early', 'mid', 'late' stages

      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            
            #? apply multiple spatial attention modules and perform maxpooling
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attetion_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attetion_module(X[m][stage])), dim=1)
            
            #max pooling over all channels
            X_att = torch.max(X_att, dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
            X[m][stage] = X_att * X[m][stage] #?apply attention to features
         
            #?patch embeddings
            X[m][stage] = self.patch_embed[m][stage](X[m][stage])
         
      #?concatenate each modality and stage
      X_fused = torch.zeros(rgb_input.size(0), 0, self.Ct).to(X['rgb']['late'].device) 
      for stage in self.stages:
         X_fused = torch.cat((X_fused, torch.cat((X['rgb'][stage], X['depth'][stage]), dim=1)), dim=1)
         
      #?prepend [cls] token as learnable parameter
      cls_tokens = self.cls_token.expand(X_fused.size(0), -1, -1) # (batch_size, 1, C)
      X_fused = torch.cat((cls_tokens, X_fused), dim=1) # (batch_size, H*W+1, C)
      #?add positional embedding as learnable parameters to each element of the sequence
      X_fused = X_fused + self.pos_embed #(batch_size, H*W+1, C)

      X_fused = self.transformer_encoder(X_fused) #transformer (with batch_first=True) expects input: #? [batch_size, sequence_length= H*W+1, dimension=C]
      
      #?classification
      cls_output = X_fused[:, 0]  #?Extract [cls] token's output
      logits = self.fc(cls_output)
      return logits, {'late': cls_output} 
        
   

class FusionNet_NOTRANSF_onlylate(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(FusionNet_NOTRANSF_onlylate, self).__init__()      
      num_classes = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #? Heights and Widths of the feature maps at different stages 
      self.stages = {'late': [352, 9]} # only late features


      self.n_spatial_attentions = 5  
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })
      
      self.conv = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.Conv2d(self.stages[stage][0], 352, kernel_size=1) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.Conv2d(self.stages[stage][0], 352, kernel_size=1) for stage in self.stages})
      })
      
      self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
      self.fc = nn.Linear(352*2, num_classes)

   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)
      
      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}  # Assume X_rgb[1] and X_depth[1] contain 'early', 'mid', 'late' stages

      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            
            #? apply multiple spatial attention modules and perform maxpooling
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attetion_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attetion_module(X[m][stage])), dim=1)
            
            #max pooling over all channels
            X_att = torch.max(X_att, dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
            X[m][stage] = X_att * X[m][stage] #?apply attention to features
            
            X[m][stage] = self.conv[m][stage](X[m][stage]) #? [batch_size, 352]
            
      #?concatenate each modality and stage
      X_fused = torch.cat((X['rgb']['late'], X['depth']['late']), dim=1)
         
      X_fused = self.maxpool(X_fused) #? [batch_size, 1, C]
      X_fused = torch.flatten(X_fused, 1)
      logits = self.fc(X_fused)
      
      return logits, {'late': X_fused}




class FusionNet_NOTRANSF_earlymidlate(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(FusionNet_NOTRANSF_earlymidlate, self).__init__()      
      num_classes = utils.utils.get_domains_and_labels(args)
      
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      
      #? Heights and Widths of the feature maps at different stages 
      self.stages = {'early': [32, 130], 'mid': [88, 17], 'late': [352, 9]}
      
      self.patch_sizes = { 'early': 13 , 'mid': 8, 'late': 1}
      self.n_spatial_attentions = 5
      self.patch_size = {}
        
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })
      
      self.conv = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.Conv2d(self.stages[stage][0], 352, kernel_size=1) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.Conv2d(self.stages[stage][0], 352, kernel_size=1) for stage in self.stages})
      })
            
      self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
      self.fc1 = nn.Linear(352*6, 352*3)
      self.fc2 = nn.Linear(352*3, 352)
      self.fc3 = nn.Linear(352, num_classes)
      
   def forward(self, rgb_input, depth_input):
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)
      
      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}  # Assume X_rgb[1] and X_depth[1] contain 'early', 'mid', 'late' stages

      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            
            #? apply multiple spatial attention modules and perform maxpooling
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attetion_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attetion_module(X[m][stage])), dim=1)
            
            #max pooling over all channels
            X_att = torch.max(X_att, dim=1, keepdim=True)[0] #? [batch_size, 1, H, W]
            
            X[m][stage] = X_att * X[m][stage] #?apply attention to features
            
            #?maxpool and project to 352
            X[m][stage] = self.conv[m][stage](X[m][stage]) #? [batch_size, 352]
            X[m][stage] = self.maxpool(X[m][stage]) #? [batch_size, 1, C]
            X[m][stage] = X[m][stage].squeeze() #? [batch_size, C]
            
      #?concatenate each modality and stage
      X_fused = torch.zeros(rgb_input.size(0), 0).to(X['rgb']['late'].device) 
      for stage in self.stages:
         X_fused = torch.cat((X_fused, torch.cat((X['rgb'][stage], X['depth'][stage]), dim=1)), dim=1)
               
      X_fused =  F.relu(self.fc1(X_fused))
      X_fused = F.relu(self.fc2(X_fused))
      logits = self.fc3(X_fused)
      
      return logits, {'late': X_fused}
   
   

class FusionNetWithPretrainedViT(nn.Module):
   def __init__(self, rgb_model, depth_model):
      super(FusionNetWithPretrainedViT, self).__init__()

      num_classes = utils.utils.get_domains_and_labels(args)
      self.rgb_model = rgb_model
      self.depth_model = depth_model
      self.pretrained_vit = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

      # Extract specific layers of the ViT encoder
      self.vit_layers = nn.ModuleList(self.pretrained_vit.encoder.layer[:-4])

      # Update the number of tokens and dimensions based on stages
      self.patch_size = {'early': 13, 'mid': 8, 'late': 1}
      #self.stages = {'late': [352, 9]}  # Adjust this according to the stages you want to use
      self.stages = {'early': [32, 130], 'mid': [120, 17], 'late': [352, 9]}
      
      self.n_spatial_attentions = 3
      self.SpatialAttentionModules = nn.ModuleDict({
         'rgb': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages}),
         'depth': nn.ModuleDict({stage: nn.ModuleList([SpatialAttentionModule(self.stages[stage][0]) for _ in range(self.n_spatial_attentions)]) for stage in self.stages})
      })

      # Replace custom patch embedding with one suited for your feature map size
      self.patch_embed = nn.ModuleDict({
      'rgb': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_size[stage], self.stages[stage][0]) for stage in self.stages}),
      'depth': nn.ModuleDict({stage: PatchEmbedding(self.stages[stage][1], self.patch_size[stage], self.stages[stage][0]) for stage in self.stages})
      })

      # Final classification head
      self.fc = nn.Linear(768, num_classes)

   def forward(self, rgb_input, depth_input):
      # Extract CNN features
      X_rgb = self.rgb_model(rgb_input)
      X_depth = self.depth_model(depth_input)

      # Attention and patch embedding
      X = {'rgb': X_rgb[1], 'depth': X_depth[1]}
      for m in ['rgb', 'depth']:          
         for stage in self.stages:
            X_att = torch.zeros(X[m][stage].size(0), 0, X[m][stage].size(2), X[m][stage].size(3)).to(X[m][stage].device)
            for spatial_attention_module in self.SpatialAttentionModules[m][stage]:
               X_att = torch.cat((X_att, spatial_attention_module(X[m][stage])), dim=1)
            X_att = torch.max(X_att, dim=1, keepdim=True)[0]
            X[m][stage] = X_att * X[m][stage]
            X[m][stage] = self.patch_embed[m][stage](X[m][stage])

      # Concatenate features from both modalities and stages
      X_fused = torch.zeros(rgb_input.size(0), 0, 768).to(X['rgb']['late'].device) 
      for stage in self.stages:
        X_fused = torch.cat((X_fused, (X['rgb'][stage] + X['depth'][stage])), dim=1)

      # Pass through selected ViT layers
      for layer in self.vit_layers:
        X_fused = layer(X_fused)[0]  # [batch_size, num_patches, Ct]

      # Classification using the [CLS] token
      cls_output = X_fused[:, 0]  # [CLS] token output
      logits = self.fc(cls_output)

      return logits, {'late': cls_output}