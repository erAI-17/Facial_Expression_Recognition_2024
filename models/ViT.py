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

        # #? Freeze all layers except the classification head
        # for param in self.model.vit.parameters():
        #     param.requires_grad = False
        # #? Optionally, you can unfreeze some specific layers if needed
        # # for param in model.vit.encoder.layer[-1].parameters():
        # #     param.requires_grad = True
        
        #? Modify the input layer to accept 4 channels
        # self.model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
        #     in_channels=4, 
        #     out_channels=self.model.vit.embeddings.patch_embeddings.projection.out_channels, 
        #     kernel_size=self.model.vit.embeddings.patch_embeddings.projection.kernel_size, 
        #     stride=self.model.vit.embeddings.patch_embeddings.projection.stride
        # )
        
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
        features = outputs.last_hidden_state[:, 0, :] #[batch_size, 768]
        return features    

    def forward(self, x):
        
        #?Concatenate RGB and depth images along the channel dimension
        #combined_input = torch.cat((x['RGB'], x['DEPTH']), dim=1)
        
        rgb = x['RGB']
        # Replicate the single channel to create a 3-channel image (1s to leave inalterated those dimensions)
        #!depth  = x['DEPTH'].repeat(1, 3, 1, 1)
        
        # rgb = self._preprocessing_(rgb)
        #! depth = self._preprocessing_(depth)
        
        # Extract features
        rgb_feat = self._extract_features_(rgb) # [32, 768]
        #!depth_feat = self._extract_features_(depth) # [32, 768]
        
        # Concatenation fusion
        #!combined_feat = torch.cat((rgb_feat, depth_feat), dim=1)
        
        #! Classification
        #!x = self.fc(combined_feat)

        return x, {'late_feat': rgb_feat}

