import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification



class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
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
                
        #?The  AutoImageProcessor  is used to preprocess input images so that they are in the correct format for the Vision Transformer (ViT) model. 
        #? This preprocessing typically includes resizing, normalization, and converting images to tensors.
    
        # #? Freeze all parameters initially
        # for param in self.model.parameters():
        #     param.requires_grad = False
            
        # #?unfreeze last classifier    
        for name, param in self.model.named_parameters():
            if 'classifier' in name: 
                param.requires_grad = True
            else: 
                param.requires_grad = False
                
        # #? Unfreeze some specific layers if needed
        # for param in self.model.encoder.layer[-1].parameters():
        #     param.requires_grad = True
                
    # Function to preprocess a single image
    def _preprocessing_(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"]

    # Function to extract features from the [CLS] token of the last hidden state.
    def _extract_features_(self, inputs):
        outputs = self.model(inputs)
        # Extract features from the [CLS] token
        features = outputs.last_hidden_state[:, 1:, :] #[batch_size, 196, 768]
        return features    

    def forward(self, x):        
        #call processor
        #x = self._preprocessing_(x)
        
        # Extract features
        x = self._extract_features_(x) #[batch_size, 196, 768]
        
        return {'late_feat': x}

