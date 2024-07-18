from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.args import args
        
ImageNet_mean = [0.485, 0.456, 0.406] 
ImageNet_std = [0.229, 0.224, 0.225]

ViT1_mean = [0.5, 0.5, 0.5]
ViT1_std = [0.5, 0.5, 0.5]  
   
class ScaleToUnitInterval_depth():
    def __call__(self, img):
        return img / 65535 #uint16
    
class ScaleToUnitInterval_rgb():
    def __call__(self, img):
        return img / 255
    
class Tofloat32():
    def __call__(self, img):
        return img.to(torch.float32)

class StackChannels():
    def __call__(self, img):
        return img.repeat(3, 1, 1)
    
class RGB_transf:
    def __init__(self, augment=False):
        init_transformations = [
            transforms.ToTensor(),  # Converts the image to a tensor but doesn't normalize to [0,1]
            Tofloat32(),
            ScaleToUnitInterval_rgb(),  # Scale to [0,1]
        ]
        normalization = [
            transforms.Normalize(mean=ViT1_mean, std=ViT1_std)  # Normalize the tensor to [-1,1]
        ]
        
        augmentations = []
        if augment:
            augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #simulates variations in lightening
            #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #simulates out of focus
            #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0)) #simulates occlusions
            ]
   
        resizing = []    
        if args.models['RGB'].model == 'RGB_efficientnet_b3':
            resizing = [transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC)]
        
        
        transformations = init_transformations + augmentations + resizing + normalization      
        self.transform = transforms.Compose(transformations)
    
    def __call__(self, img):
        # Convert NumPy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Apply transformations
        img = self.transform(img)
        
        return img
    
class DEPTH_transf:
    def __init__(self, augment=False):    
        
        init_transformations = [
            transforms.ToTensor(),  # Converts the image to a tensor but doesn't normalize to [0,1]
            Tofloat32(),
            StackChannels(), #stack the single channel into 3 channels
            ScaleToUnitInterval_depth(), #need to scale into [0,1]
        ]
        
        normalization = [
            transforms.Normalize(mean=ImageNet_mean, std=ImageNet_std)
        ]
        
        augmentations = []
        if augment:
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0))
            
            ]
            
        resizing = []
        if args.models['DEPTH'].model == 'DEPTH_efficientnet_b3':
            resizing = [transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC)]
            transformations = resizing + transformations
        
        transformations = init_transformations + augmentations + resizing + normalization  
        self.transform = transforms.Compose(transformations)
    
    def __call__(self, img):
        # Convert NumPy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Apply transformations
        img = self.transform(img)
        
        return img    
