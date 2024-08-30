from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from utils.args import args
import cv2
#import mediapipe as mp #!only for online alignment

        
ImageNet_mean = [0.485, 0.456, 0.406] 
ImageNet_std = [0.229, 0.224, 0.225]

CalD3rMenD3s_mean_RGB = [0.4954188046463236, 0.365289621256328, 0.3194358753260218]
CalD3rMenD3s_std_RGB = [0.26393761653339637, 0.21517540134099997, 0.21502950740118704]
CalD3rMenD3s_mean_DEPTH = [0.3601669753802989, 0.3601669753802989, 0.3601669753802989]
CalD3rMenD3s_std_DEPTH = [0.07899400881961408, 0.07899400881961408, 0.07899400881961408]

class ToTensorUint16:
    def __call__(self, img): #receives np array float32 
        img_np = np.array(img)
        
        if args.dataset.name == 'CalD3rMenD3s': 
            img_np = img_np / 9785.0 #(originally was uint16 ranging in 0-9785)
        elif args.dataset.name == 'BU3DFE':
            img_np = img_np / 65535.0
        
        # If the image is grayscale, add a channel dimension
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
            img_np = np.repeat(img_np, 3, axis=-1)
        
        # Convert to PyTorch tensor float32
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(torch.float32)
        
        return img_tensor
    
class Hysto_Eq():
    def __init__(self, grayscale=False):
        self.grayscale = grayscale
       
    def __call__(self, img):
        if self.grayscale:
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.equalizeHist(gray_img)
            
            # Stack the grayscale array to create a 3-channel image
            img = np.stack((gray_img,)*3, axis=-1)
        
        else:
            # Convert the image to YCrCb color space
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

            # Equalize the histogram of the Y channel
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

            # Convert back to RGB color space
            img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
        
        return img    

class Transform:
    def __init__(self, augment=False, mean=None, std=None):
        
        self.hystr_eq = {
            'RGB': [
                Hysto_Eq(grayscale=False)
            ],
            'DEPTH': []
        }
        
        self.to_tensor = {
            'RGB': [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ],
            'DEPTH': [
                ToTensorUint16(),
            ]
        }
        
        self.resize = []
        if args.models['RGB'].model == 'efficientnet_b2' and args.models['DEPTH'].model == 'efficientnet_b2':
            self.resize = [transforms.Resize((260, 260), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        if args.models['RGB'].model == 'mobilenet_v4' and args.models['DEPTH'].model == 'mobilenet_v4':
            self.resize = [transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        
        self.augment = []
        if augment:
            self.augment = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0)),
            ]
            
        self.normalize = {
            'RGB': [
                transforms.Normalize(mean['RGB'], std['RGB']),
            ],
            'DEPTH': [mean['DEPTH'], std['DEPTH']]
        }
        
        #! compose transformations
        self.rgb_transformations = self.hystr_eq['RGB'] + self.to_tensor['RGB'] + self.resize + self.augment + self.normalize['RGB']
        self.RGB_transform = transforms.Compose(self.rgb_transformations)
        
        self.depth_transformations = self.to_tensor['DEPTH'] + self.resize + self.augment + self.normalize['DEPTH']
        self.DEPTH_transform = transforms.Compose(self.depth_transformations)
    
    def __call__(self, sample):       
        img = np.array(sample['RGB'])
        depth = np.array(sample['DEPTH'])
        
        # Apply the same augmentations to both
        seed = np.random.randint(2147483647) 

        # Apply transformations
        torch.manual_seed(seed)
        img = self.RGB_transform(img)
        
        torch.manual_seed(seed)
        depth = self.DEPTH_transform(depth)
        
        sample = {'RGB': img, 'DEPTH': depth}
        
        return sample
    

    

