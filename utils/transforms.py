import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from utils.args import args

#!###
#! TRANFORMATIONS
#!##
class ToFloat32:
        def __call__(self, x):
            return x.to(torch.float32)
        
class ScaleToUnitInterval():
    def __call__(self, img):
        return img / img.max()
    
class RGB_transf:
    def __init__(self, augment=False):
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                transforms.RandomRotation(10),  # Randomly rotate the image by 10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize the image
                transforms.ToTensor(),  # Converts the image to a tensor but doesn't normalize to [0,1]
                ToFloat32(),  # Ensures the tensor is of type float32
                ScaleToUnitInterval(),  # Scale to [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor to [-1,1]
            ])
        else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts the image unit8 [0,255] to a tensor and normalizes to [0,1]
                    ToFloat32(),  # Ensures the tensor is of type float32
                    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor to [-1,1]
                ])
    
    def __call__(self, img):
        # Convert NumPy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Apply transformations
        img = self.transform(img)
        
        return img
    
class DEPTH_transf:
    def __init__(self, augment=False):
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop and resize the image
                transforms.ToTensor(),  # Converts the image to a tensor BUT it doesn't noralize into [0,1] because input image is unit16 [0,65535]
                ToFloat32(),  # Ensures the tensor is of type float32
                ScaleToUnitInterval(),# need to scale into [0,1] because .toTensor() doesn't automatically do it
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor to [-1,1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts the image to a tensor BUT it doesn't noralize into [0,1] because input image is unit16 [0,65535]
                ToFloat32(),  # Ensures the tensor is of type float32
                ScaleToUnitInterval(),# need to scale into [0,1] because .toTensor() doesn't automatically do it
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor to [-1,1]
            ])
    
    def __call__(self, img):
        # Convert NumPy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Apply transformations
        img = self.transform(img)
        
        return img    



if __name__ == "__main__":
    #!##
    #!TRY different configurations
    #!##
    rgb_tranf = RGB_transf()
    depth_tranf = DEPTH_transf()

    img = cv2.imread('../Datasets/CalD3r/Anger/RGB/F_001_1120_anger_Color.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = cv2.imread('../Datasets/CalD3r/Anger/DEPTH/F_001_1120_anger_Depth.png', cv2.IMREAD_UNCHANGED)
    #? img: (224,224,3) values [0,255] unint8
    #? depth: (224, 224) values [0,4868] unint16 ([0,19] without cv2.IMREAD_UNCHANGED) 
    
    transf_img = rgb_tranf(img)
    
    transf_depth = depth_tranf(depth)
    #? transf_img: torch.Size(3, 224, 224) values [-1,1] float32
    #? transf_depth: torch.Size(1, 224,224) values [-1,1] float32
    
    transf_max= transf_depth.max()
    transf_min= transf_depth.min()
    
    plt.imshow(depth)
    plt.axis('off')  # Hide axes
    plt.show()

    plt.imshow(transf_img.permute(1,2,0))
    plt.axis('off')  # Hide axes
    plt.show()

    plt.imshow(transf_depth.permute(1,2,0))
    plt.axis('off')  # Hide axes
    plt.show()