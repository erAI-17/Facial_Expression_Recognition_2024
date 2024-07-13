from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.args import args
        
ImageNet_mean = [0.485, 0.456, 0.406] 
ImageNet_std = [0.229, 0.224, 0.225]

ViT1_mean = [0.5, 0.5, 0.5]
ViT1_std = [0.5, 0.5, 0.5]  
   
class ScaleToUnitInterval():
    def __call__(self, img):
        return img / 255#img.max()
    
class Tofloat32():
    def __call__(self, img):
        return img.to(torch.float32)

class StackChannels():
    def __call__(self, img):
        return img.repeat(3, 1, 1)
    
class RGB_transf:
    def __init__(self, augment=False):
        transformations = [
            transforms.ToTensor(),  # Converts the image to a tensor but doesn't normalize to [0,1]
            ScaleToUnitInterval(),  # Scale to [0,1]
            transforms.Normalize(mean=ViT1_mean, std=ViT1_std)  # Normalize the tensor to [-1,1]
        ]
        if augment:
            augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            ]
            transformations = augmentations + transformations
            
        if args.models['RGB'].model == 'RGB_efficientnet_b3':
            resizing = [transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC)]
            transformations = resizing + transformations
            
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
        transformations = [
            ScaleToUnitInterval(), #need to scale into [0,1]
            transforms.Normalize(mean=ImageNet_mean, std=ImageNet_std)
        ]
        
        if augment:
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
            transformations = augmentations + transformations
            
        if args.models['DEPTH'].model == 'DEPTH_efficientnet_b3':
            resizing = [transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC)]
            transformations = resizing + transformations
        
        stack_cannels = [
            transforms.ToTensor(), 
            Tofloat32(),
            StackChannels(), #stack the single channel into 3 channels
        ]
        transformations = stack_cannels + transformations
        
        self.transform = transforms.Compose(transformations)
    
    def __call__(self, img):
        # Convert NumPy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Apply transformations
        img = self.transform(img)
        
        return img    
