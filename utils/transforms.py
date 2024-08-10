from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from utils.args import args
import cv2
        
ImageNet_mean = [0.485, 0.456, 0.406] 
ImageNet_std = [0.229, 0.224, 0.225]

Calder_Mendes_mean_RGB = [0.49102524485829907, 0.3618844398451173, 0.31640123102109985]
Calder_Mendes_std_RGB = [0.26517980798288976, 0.21546631829305746, 0.21493371251079485]
Calder_Mendes_mean_DEPTH = [0.3581094589547684, 0.3581094589547684, 0.3581094589547684]
Calder_Mendes_std_DEPTH = [0.08069561050494341, 0.08069561050494341, 0.08069561050494341]

class ToTensorUint16:
    def __call__(self, img): #receives PIL image in uint16
        # Convert image to numpy array
        img_np = np.array(img).astype(np.float32)
        
        # Scale the image to [0, 1] by dividing by 9785
        img_np = img_np / 9785.0
        
        # If the image is grayscale, add a channel dimension
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
            img_np = np.repeat(img_np, 3, axis=-1)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor
        
        
class RGBTransform:
    def __init__(self, augment=False):
        self.to_tensor = [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), #convert to float32 and scale to [0,1] (deviding by 255)
        ]
        
        self.resize = []    
        if args.models['RGB'].model == 'efficientnet_b2':
            self.resize = [transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
            
        self.augment = []
        if augment:
            self.augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #simulates out of focus
            #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0)), #simulates occlusions
            ]
            
        self.normalize = [
            transforms.Normalize(mean=Calder_Mendes_mean_RGB, std=Calder_Mendes_std_RGB),  # Normalize the tensor to [-1,1]
        ]
        
        self.transformations = self.to_tensor + self.resize + self.augment + self.normalize    
        self.transform = transforms.Compose(self.transformations)
            
    def __call__(self, img):            
        # Apply transformations
        img = self.transform(img)
        
        return img
    
class DEPTHTransform:
    def __init__(self, augment=False):    
        
        self.to_tensor = [
            ToTensorUint16(),  # Converts the image to a tensor but doesn't normalize to [0,1]
        ]    
        
        self.resize = []
        if args.models['DEPTH'].model == 'efficientnet_b2':
            self.resize = [transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BILINEAR),
            ]
            
        self.augment = []
        if augment:
            self.augment = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0)),
            ]
        
        self.normalize = [
            transforms.Normalize(mean=Calder_Mendes_mean_DEPTH, std=Calder_Mendes_std_DEPTH),
        ]
        
        self.transformations = self.to_tensor + self.resize + self.augment + self.normalize    
        self.transform = transforms.Compose(self.transformations)
    
    def __call__(self, img):            
        # Apply transformations
        img = self.transform(img)
        
        return img    
