import torch
from torch import nn
import torch.nn.functional as F
import utils
from utils.args import args
import torchaudio.transforms as T
from transformers import AutoImageProcessor, AutoModel



#?The  AutoImageProcessor  is used to preprocess input images so that they are in the correct format for the Vision Transformer (ViT) model. 
#? This preprocessing typically includes resizing, normalization, and converting images to tensors. 
processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

#? Load pre-trained ViT model
model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

# Freeze all layers except the classification head
for param in model.vit.parameters():
    param.requires_grad = False

# Optionally, you can unfreeze some specific layers if needed
# for param in model.vit.encoder.layer[-1].parameters():
#     param.requires_grad = True

# Function to preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

# Function to extract features from the [CLS] token of the last hidden state.
# These features can then be used for various downstream tasks such as classification, clustering, or further processing in a multi-modal fusion network.
def extract_features(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    # Extract features from the [CLS] token
    features = outputs.last_hidden_state[:, 0, :]
    return features
