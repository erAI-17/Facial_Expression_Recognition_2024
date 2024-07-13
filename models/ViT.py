import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification



class ViT(nn.Module):
    def __init__(self, p_dropout):
        super(ViT, self).__init__()
        
        self.p_dropout = p_dropout
        
        #? Load pre-trained ViT model
        #!1 FER2013,MMI Facial Expression Database, and AffectNet Accuracy: 0.8434 - Loss: 0.4503 (454d)
        self.model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
        self.processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
        
        #!2 FER2013 Accuracy - 0.922 Loss - 0.213 (11d)
        # outer_model = AutoModelForImageClassification.from_pretrained("HardlyHumans/Facial-expression-detection")
        # self.model = outer_model.vit
        # self.processor = AutoImageProcessor.from_pretrained("HardlyHumans/Facial-expression-detection")
        
        #!3 FER2013 Accuracy: 0.7113 (412,838d)
        # outer_model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
        # self.model = outer_model.vit
        # self.processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        
        #? Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
                            
        #? Unfreeze last 2 encoder layers, the "layernorm" and "pooler" 
        # for i in range(10, 12):
        #     for param in self.model.encoder.layer[i].parameters():
        #         param.requires_grad = True
        for param in self.model.pooler.parameters():
            param.requires_grad = True
        for param in self.model.layernorm.parameters():
            param.requires_grad = True
        
        #? set dropout
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.p_dropout

    def forward(self, x):        
        #x = self._preprocessing_(x)
        
        x = self.model(x) #? [batch_size, 197, 768]
        
        # Extract features
        late_feat = x.last_hidden_state[:, 1:, :]
        
        # Extract cls token
        cls = x.last_hidden_state[:, :1, :].squeeze() #? [batch_size, 768]
        
        return cls, {'late_feat': late_feat}

