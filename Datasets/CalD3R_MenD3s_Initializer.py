import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
import PIL.Image as Image
import mediapipe as mp

def train_test_annotations():
    """Splits dataset into TRAIN and TEST splits and generate annotation .pkl files where a row represents a sample with this schema:
    dataset (str): CalD3r, MenD3s
    subj_id (str): unique code
    code (str): same subj_id for same label, may have multiple samples
    label (str): anger, surprise,...
    add (list(str, str)): list storing additional info ( gender, pose,...) 
    """
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    datasets = ['CalD3r', 'MenD3s']
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    class_distribution = {'Color':{emotion: 0 for emotion in emotions.keys()}, 'Depth':{emotion: 0 for emotion in emotions.keys()}}
    
    # Initialize accumulators for mean and std
    mean = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    std = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_sq_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    n_pix = {'Color': 0, 'Depth': 0}
    data = []
    
    max_depth = 0
    for dataset in datasets:
        path = f'../Datasets/CalD3RMenD3s/{dataset}'
        
        for emotion in emotions.keys(): 
            for m in ['Color', 'Depth']:
                
                if m == 'Color':
                    files_path = f'{path}/{emotion.capitalize()}/RGB'
                else:
                    files_path = f'{path}/{emotion.capitalize()}/DEPTH'
                
                for filename in os.listdir(files_path):
                    #remove "aligned_" from filename
                    info_filename = filename.replace('aligned_', '')
                    add = [info_filename.split("_")[0]] 
                    subj_id = info_filename.split("_")[1]   
                    code = info_filename.split("_")[2]
                    description_label = info_filename.split("_")[3]
                    
                    
                    new_entry = [dataset, subj_id, code, description_label, emotions[description_label], add]
                    if new_entry not in data: #avoid duplicates (same sample with different modalities)
                        data.append([dataset, subj_id, code, description_label, emotions[description_label], add])

                    #!update class distribution
                    class_distribution[m][description_label] += 1
                  
                    #!load image
                    img_path = f'{files_path}/{filename}'
                    if m == 'Color':
                        img =  Image.open(img_path)
                        img = np.array(img)
                         
                        #Resize images to input size of the model you will use
                        img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
                        
                        # Normalize to [0, 1]
                        img = img / 255.0  
                        
                        #!update mean and std
                        sum_pix[m] += np.sum(img, axis=(0, 1)) #sum all pixels in the image, separately for each channel (black pixels are 0)
                        sum_sq_pix[m] += np.sum(img ** 2, axis=(0, 1))
                       
                        # Create a mask to maintain only pixels where all three channels are below the threshold
                        mask = (img[:, :, 0] > 0) | (img[:, :, 1] >  0) | (img[:, :, 2] > 0)
                        #mask off 0 values (black pixels) in the frame, from each channel
                        img = img[mask]
                        n_pix[m] += mask.sum() #mask converts into a 2D array
                        
                    elif m == 'Depth':
                        img = Image.open(img_path)
                        img = np.array(img)
                        #!convert to 3 channels                       
                        img = np.expand_dims(img, axis=-1)
                        img = np.repeat(img, 3, axis=-1)
                        
                        max_depth = max(max_depth, img.max()) #!get max depth before normalization
                        
                        #Resize images to input size of the model you will use
                        img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
                        
                        # Normalize to [0, 1] using max_depth=9785
                        img = img / 9785.0  
                        
                        #!update mean and std
                        sum_pix[m] += np.sum(img, axis=(0, 1))
                        sum_sq_pix[m] += np.sum(img ** 2, axis=(0, 1))
                        
                        # Create a mask for each channel where pixel values are greater than the threshold=0 (to avoid balck frame pixels)
                        mask = (img[:, :, 0] > 0) | (img[:, :, 1] >  0) | (img[:, :, 2] > 0)
                        #mask off 0 values (black pixels) in the frame, from each channel
                        img = img[mask]
                        n_pix[m] += mask.sum() #mask converts into a 2D array
                                        
    # Average mean and std over the number of samples
    for m in ['Color', 'Depth']:
        mean[m] = sum_pix[m] / n_pix[m]
        std[m] = np.sqrt(sum_sq_pix[m] / n_pix[m] - mean[m] ** 2)

    #convert to dataframes
    complete_df = pd.DataFrame(data, columns=['dataset','subj_id', 'code', 'description_label', 'label', 'add'])

    #save annotation test file
    annotation_file = os.path.join('../Datasets/CalD3RMenD3s/', 'annotations_complete.pkl')
    with open(annotation_file, 'wb') as file:
        pickle.dump(complete_df, file)
    
    return class_distribution, mean, std 
  

def align_face():
    # Initialize the face alignment object
    alignment = Alignment()
    datasets = ['CalD3r', 'MenD3s']
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    for dataset in datasets:
        path = f'../Datasets/Original_CalD3RMenD3s/{dataset}'
        for emotion in emotions.keys(): 
            file_path = f'{path}/{emotion.capitalize()}/RGB'
            for filename in os.listdir(file_path): 
                #!align images and save
                img =  Image.open(f'{file_path}/{filename}')
                try: 
                    depth = Image.open(f'{path}/{emotion.capitalize()}/DEPTH/{filename.replace('_Color', '_Depth')}')
                except:
                    print(f'No depth image for {filename}')
                    continue
                
                try:
                    img_al, depth_al = alignment(img, depth, overlay=False)
                except:
                    print(f'No face detected in {filename}')
                    continue
                
                #!show aligned images
                # plt.subplot(1, 2, 1)
                # plt.imshow(img)
                # plt.axis('off')
                # plt.title('Original RGB')
                # plt.subplot(1, 2, 2)
                # plt.imshow(img_al)
                # plt.axis('off')
                # plt.title('Transformed RGB')
                # plt.show()
                
                # plt.subplot(1, 2, 1)
                # plt.imshow(depth)
                # plt.axis('off')
                # plt.title('Original depth')
                # plt.subplot(1, 2, 2)
                # plt.imshow(depth_al)
                # plt.axis('off')
                # plt.title('Transformed depth')
                # plt.show()
                
                #!save aligned images as png into directory  "CalD3RMenD3s"  !=  "Original_CalD3RMenD3s"
                save_path_rgb = f'../Datasets/CalD3RMenD3s/{dataset}/{emotion.capitalize()}/RGB/aligned_{filename}'
                save_path_depth = f'../Datasets/CalD3RMenD3s/{dataset}/{emotion.capitalize()}/DEPTH/aligned_{filename.replace('_Color', '_Depth')}'
                #save aligned images, convert from BGR to RGB
                cv2.imwrite(save_path_rgb, cv2.cvtColor(img_al, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, depth_al)
                
                print(f'Images saved at {save_path_rgb}')
                


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
        
    
#!##
#!MAIN
#!##
if __name__ == '__main__':
    

    #!align images and save
    #align_face()
    
    #!generate annotation files for each dataset, TEST and TRAIN
    class_distribution, mean, std = train_test_annotations() #20% test, 80% train
    #! plot histogram class distribution
    class_distribution = class_distribution['Color']
    plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue', alpha=0.8)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes')
    # Add values on top of each bar
    for i, (key, value) in enumerate(class_distribution.items()):
        plt.text(i, value, str(value), ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    plt.savefig('/Images/CalD3RMenD3s_distribution.png')
                
        
    
    
    
    
    
    
