import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from transformers import AutoImageProcessor, AutoModel



class ViT(nn.Module):
    def __init__(self):
        num_classes, valid_labels = utils.utils.get_domains_and_labels(args)
        super(ViT, self).__init__()
        #? Load pre-trained ViT model
        self.model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
        
        #?The  AutoImageProcessor  is used to preprocess input images so that they are in the correct format for the Vision Transformer (ViT) model. 
        #? This preprocessing typically includes resizing, normalization, and converting images to tensors. 
        self.processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

        # #? Freeze all layers except the classification head
        # for param in self.model.vit.parameters():
        #     param.requires_grad = False
        # #? Optionally, you can unfreeze some specific layers if needed
        # # for param in model.vit.encoder.layer[-1].parameters():
        # #     param.requires_grad = True
        
        #classification layer
        self.fc = nn.Linear(768 *2, num_classes)  
        
    # Function to preprocess a single image
    def _preprocessing_(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"]

    # Function to extract features from the [CLS] token of the last hidden state.
    def _extract_features_(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        # Extract features from the [CLS] token
        features = outputs.last_hidden_state[:, 0, :]
        return features    

    def forward(self, x):
        rgb_prep = self._preprocessing_(x['RGB'])
        depth_prep = self._preprocessing_(x['DEPTH'])
        print(f"RGB preprocessed shape: {rgb_prep.shape}")
        print(f"Depth preprocessed shape: {depth_prep.shape}")
        
        # Extract features
        rgb_feat = self._extract_features_(rgb_prep)
        depth_feat = self._extract_features_(depth_prep)
        print(f"RGB features shape: {rgb_feat.shape}")
        print(f"Depth features shape: {depth_feat.shape}")
        
        # Concatenation fusion
        combined_feat = torch.cat((rgb_feat, depth_feat), dim=1)
        print(f"Combined features shape: {combined_feat.shape}")
        
        # Classification
        x = self.fc(combined_feat)
        return x, {}

