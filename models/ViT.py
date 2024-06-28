import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from transformers import AutoImageProcessor, AutoModel



class ViT(nn.Module):
    """This model is a fine-tuned version of:
        google/vit-base-patch16-224-in21k 
    on the FER 2013, MMI Facial Expression Database, and AffectNet dataset datasets.
    """
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(ViT, self).__init__()
        #? Load pre-trained ViT model
        self.model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
        
        #?The  AutoImageProcessor  is used to preprocess input images so that they are in the correct format for the Vision Transformer (ViT) model. 
        #? This preprocessing typically includes resizing, normalization, and converting images to tensors. 
        self.processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

        #? Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        #?unfreeze last classifier    
        for name, param in self.model.named_parameters():
            if 'classifier' in name: 
                param.requires_grad = True
                
        #? Optionally, you can unfreeze some specific layers if needed
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
                
    # Function to preprocess a single image
    def _preprocessing_(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"]

    # Function to extract features from the [CLS] token of the last hidden state.
    def _extract_features_(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        # Extract features from the [CLS] token
        features = outputs.last_hidden_state[:, 0, :] #[batch_size, 768]
        return features    

    def forward(self, x):
        
        # Extract features
        rgb_feat = self._extract_features_(x) # [32, 768]

        return x, {'late_feat': rgb_feat}

