import cv2
import matplotlib.pyplot as plt
from vedo import Plotter, load
import vtk
from vtkmodules.util import numpy_support
import numpy as np
import os
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import shutil


def wrl_to_depth_map(path):     
    # Load the 3D model from the .wrl file
    mesh = load(path)
    
    # Create a plotter for rendering
    plotter = Plotter(offscreen=True, size=(512, 3072)) #SET TO FALSE TO visualize
    plotter.add(mesh)
    plotter.show(interactive=False) #SET TO TRUE TO VISUALIZE
    
    # Get the vtk renderer window
    window = plotter.window

    # Create a vtk window-to-image filter to capture the depth buffer
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(window)
    w2if.SetInputBufferTypeToZBuffer()  # Capture the Z-buffer
    w2if.Update()

    # Convert the depth data to a numpy array
    depth_image = w2if.GetOutput()
    width, height, _ = depth_image.GetDimensions()
    zbuffer_array = numpy_support.vtk_to_numpy(depth_image.GetPointData().GetScalars())
    zbuffer_array = zbuffer_array.reshape((height, width))

    # Normalize depth map for visualization
    zbuffer_normalized = (zbuffer_array - zbuffer_array.min()) / (zbuffer_array.max() - zbuffer_array.min())

    # Convert to 8-bit image for easy thresholding
    zbuffer_normalized_uint8 = (zbuffer_normalized * 255).astype(np.uint8)

    # Threshold to identify the white (or nearly white) areas
    _, thresholded = cv2.threshold(zbuffer_normalized_uint8, 250, 255, cv2.THRESH_BINARY) 

    # Invert thresholded image to get the actual depth areas
    thresholded_inv = cv2.bitwise_not(thresholded)

    # Find contours to detect the bounding box of the non-white area
    contours, _ = cv2.findContours(thresholded_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    # Ensure bounding box is within image dimensions
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, width - x), min(h, height - y)
    
    # Crop the depth map to the bounding box
    depth = zbuffer_array[y:y+h, x:x+w] #zbuffer_normalized #?zbuffer_array #zbuffer_normalized
    
    # Resize the cropped depth map while maintaining the aspect ratio
    h, w = depth.shape
    aspect_ratio = w / h
    new_width, new_height = 512, 512

    if aspect_ratio > 1:
        # Wider than tall
        new_width = 512
        new_height = int(512 / aspect_ratio)
    else:
        # Taller than wide
        new_height = 512
        new_width = int(512 * aspect_ratio)

    # Resize to fit within the new dimensions
    depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a 512x512 canvas and place the resized image in the center
    depth_padded = np.zeros((512, 512), dtype=np.float32)
    start_x = (512 - new_width) // 2
    start_y = (512 - new_height) // 2
    depth_padded[start_y:start_y + new_height, start_x:start_x + new_width] = depth
    
    # Convert the white background to black in the depth map
    depth_padded[depth_padded ==1] = 0.0

    # Rotate the depth map by 180 degrees
    depth_padded = cv2.rotate(depth_padded, cv2.ROTATE_180) 
    
    #vertical flip
    depth_padded = cv2.flip(depth_padded, 1)
    
    # Convert from float32 to uint16bit image
    depth_padded_uint16 = (depth_padded * 65535).astype(np.uint16)
    
    #Display the depth map #float32
    # plt.figure(figsize=(8, 6))
    # plt.imshow(depth_padded, cmap='gray')
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(depth_padded_uint16, cmap='gray')
    # plt.show()

    #Save depth map as grayscale image 
    cv2.imwrite(path[:-4] + '_depth.png', depth_padded_uint16)

    plotter.close()
    

def train_test_annotations():
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    emotions = {'AN':0, 'DI':1, 'FE':2, 'HA':3, 'NE':4, 'SA':5, 'SU':6}    
    class_distribution = {'Color':{emotion: 0 for emotion in emotions.keys()}, 'Depth':{emotion: 0 for emotion in emotions.keys()}}
    
    # Initialize accumulators for mean and std
    mean = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    std = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    sum_sq_pix = {'Color': np.zeros(3), 'Depth': np.zeros(3)}
    n_pix = {'Color': 0, 'Depth': 0}

    max_depth = 0
    path = f'../Datasets/BU3DFE/Subjects'
    data = []  
    for subject in os.listdir(path): 
        for filename in os.listdir(path + '/' + subject):
            subj_id = filename.split("_")[0]   
            label = filename.split("_")[1][:2]
            intensity = filename.split("_")[1][2:4]
            race = filename.split("_")[1][4:6]
            m = 'Depth' if filename.endswith('depth.png') else 'Color'
            
            new_entry = [subj_id, label, intensity, race, emotions[label]]
            if new_entry not in data: #avoid duplicates (same sample with different modalities)
                data.append([subj_id, label, intensity, race, emotions[label]])

            #!update class distribution
            class_distribution[m][label] += 1
            
            #!load image
            img_path = f'{path}/{subject}/{filename}'
            if m == 'Color':
                img =  Image.open(img_path)
                img = np.array(img) #[512,512,3]
                    
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
    
    #convert to dataframe
    complete_df = pd.DataFrame(data, columns=['subj_id', 'description_label' , 'intensity', 'race', 'label'])
    
    #save annotation train file
    annotation_file = os.path.join('../Datasets/BU3DFE/', 'annotations_complete.pkl')
    with open(annotation_file, 'wb') as file:
        pickle.dump(complete_df, file)
    
    return class_distribution, mean, std   

#!##
#!MAIN
#!##
if __name__ == '__main__':
    
    #!generate depth maps for all .wrl files
    # path = '../Datasets/Original_BU3DFE/Subjects/'
    # count = 0
    # for subject in os.listdir(path):
    #     for file in os.listdir(path + subject):
    #         if file.endswith('F3D.wrl'):
    #             print('Processing: ' + path + subject + '/' + file)
    #             wrl_to_depth_map(path + subject + '/' + file)
             
    # # example
    # img_path = '../Datasets/Original_BU3DFE/Subjects/F0001/F0001_AN01WH_F3D_depth.png'
    # img = Image.open(img_path)
    # img = np.array(img) #[512,512,3]
    
    
    #! copy all files ending with _F2D.bmp or F3D_depth.png to a new folder "BU3DFE"
    source_path = '../Datasets/Original_BU3DFE/Subjects/'
    dest_path = '../Datasets/BU3DFE/Subjects/'

    for subject in os.listdir(source_path):
        subject_source_path = os.path.join(source_path, subject)
        subject_dest_path = os.path.join(dest_path, subject)

        # Create the destination directory if it doesn't exist
        os.makedirs(subject_dest_path, exist_ok=True)

        for file in os.listdir(subject_source_path):
            if file.endswith('F2D.bmp') or file.endswith('F3D_depth.png'):
                source_file = os.path.join(subject_source_path, file)
                dest_file = os.path.join(subject_dest_path, file)
                print('Copying:', source_file, 'to', dest_file)
                shutil.copy2(source_file, dest_file)

    #!generate annotation files for each dataset, TEST and TRAIN
    class_distribution, mean, std = train_test_annotations() #20% test, 80% train
    #!plot histogram class distribution
    full_emot = {'AN': 'anger', 'DI': 'disgust', 'FE': 'fear', 'HA': 'happiness', 'NE': 'neutral', 'SA': 'sadness', 'SU': 'surprise'}
    class_distribution = class_distribution['Color']
    full_class = [full_emot[emotion] for emotion in class_distribution.keys()]
    plt.bar(full_class, class_distribution.values(), color='skyblue', alpha=0.8)
    #Set the y-axis limit
    plt.ylim(0, 800)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes')
    #Add values on top of each bar
    for i, (key, value) in enumerate(class_distribution.items()):
        plt.text(i, value, str(value), ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    plt.savefig('/Images/BU3DFE_distribution.png')