from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from utils.args import args
import cv2
#import mediapipe as mp #!only for online alignment

        
ImageNet_mean = [0.485, 0.456, 0.406] 
ImageNet_std = [0.229, 0.224, 0.225]

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
    def __init__(self, augment, mean=None, std=None):
        
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
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomErasing(scale=(0.02, 0.25), ratio=(0.5, 2.0)),
            ]
            
        self.normalize = {
            'RGB': [
                transforms.Normalize(mean['RGB'], std['RGB']),
            ],
            'DEPTH': [
                transforms.Normalize(mean['DEPTH'], std['DEPTH']),
            ]
        }
        
        #! compose transformations
        self.rgb_transformations = self.hystr_eq['RGB'] + self.to_tensor['RGB'] + self.resize + self.augment + self.normalize['RGB']
        self.RGB_transform = transforms.Compose(self.rgb_transformations)
        
        self.depth_transformations = self.to_tensor['DEPTH'] + self.resize + self.augment + self.normalize['DEPTH']
        self.DEPTH_transform = transforms.Compose(self.depth_transformations)
    
    def __call__(self, sample):       
        img = np.array(sample['RGB'])
        depth = np.array(sample['DEPTH'])
        
        # if args.align_face:
        #     img, depth = Alignment()(img, depth)
        
        # Apply the same augmentations to both
        seed = np.random.randint(2147483647) 

        # Apply transformations
        torch.manual_seed(seed)
        img = self.RGB_transform(img)
        
        torch.manual_seed(seed)
        depth = self.DEPTH_transform(depth)
        
        sample = {'RGB': img, 'DEPTH': depth}
        
        return sample
    
    

class Alignment():
    def __init__(self):
        # Initialize mp_face_mesh within the worker process
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def _landmark_extraction(self, img):
        # Process the image
        results = self.mp_face_mesh.process(img)
        
        # Extract the landmarks from the first face detected
        landmarks = results.multi_face_landmarks[0].landmark
        
        return landmarks
    
    def landmark_overlay(self, img, landmarks):
        # Draw the landmarks on the image
        for landmark in landmarks:
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
                
        return img
    
    def face_alignment(self, img, depth, landmarks):
        #landmarks into image coordinates
        h, w = img.shape[:2]
        landmarks = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks])
        
       # coordinates of the corners of the eyes
        left_eye_corners_idx = [33, 133]
        right_eye_corners_idx = [362, 263]
        
        #average coordinates of corners of the eyes to get the center
        left_eye_center = np.mean([landmarks[i] for i in left_eye_corners_idx], axis=0)
        right_eye_center = np.mean([landmarks[i] for i in right_eye_corners_idx], axis=0)
        
        
        #calculate the angle between eye centers
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi
        
        #calculate the center between the eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
        
        #calculate the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
        
        #rotate the image and depth map
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return img, depth
    
    def __call__(self, img, depth, overlay=False):
        img = np.array(img)
        depth = np.array(depth)
        
        landmarks = self._landmark_extraction(img)
        
        if overlay:
            img = self.landmark_overlay(img, landmarks)
        
        img, depth = self.face_alignment(img, depth, landmarks)
        
        return img, depth

    

