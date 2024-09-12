import numpy as np
import os
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from PIL import Image, ImageOps
import cv2
import PIL.Image as Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from skimage.transform import resize
import pickle
import pandas as pd
import shutil
import mediapipe as mp


def convert_AUs_to_labels(path):
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    for subject in os.listdir(path): 
        for filename in os.listdir(path + '/' + subject):
            extension = os.path.splitext(filename)[1]
            source_file = path + '/' + subject + '/' + filename
            dest_path = os.path.join('../Datasets/Bosphorus/Subjects/', subject)
            os.makedirs(dest_path, exist_ok=True)
            
            #! if neutral
            if filename.split('_')[1] == 'N' and (extension == '.png' or extension == '.bnt') and (len(filename.split('_')) <= 4):
                shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] + '_NEUTRAL_0' + extension)
            
            #! already annotated in class
            if filename.split('_')[1] == 'E' and (extension == '.png' or extension == '.bnt') and (len(filename.split('_')) <= 4):
                shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] + '_' + filename.split('_')[2] + '_0' + extension)
            
            #conversion
            if filename.split('_')[1] == 'LFAU' and (extension == '.png' or extension == '.bnt') and (len(filename.split('_')) <= 4):
                #!anger
                
                if filename.split('_')[2] == '24':
                    #copy into new folder and change name
                    shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] +'_ANGER' + '_1' + extension)

                
                #!disgust
                if filename.split('_')[2] == '9':
                    shutil.copyfile(source_file, dest_path +'/' + filename.split('_')[0] +'_DISGUST' + '_1' + extension)
           
                #!happiness
                if filename.split('_')[2] == '12':
                    shutil.copyfile(source_file, dest_path + '/'+ filename.split('_')[0] +'_HAPPY' + '_1' + extension)
                    
                #!sadness
                if filename.split('_')[2] == '15':
                    shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] +'_SADNESS' + '_1' + extension)
                
            if filename.split('_')[1] == 'UFAU' and (extension == '.png' or extension == '.bnt') and (len(filename.split('_')) <= 4):
                #!fear
                if filename.split('_')[2] == '1':
                    shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] +'_FEAR' + '_1'+ extension)
                
                #!surprise
                if filename.split('_')[2] == '2':
                    shutil.copyfile(source_file, dest_path + '/' + filename.split('_')[0] +'_SURPRISE' + '_1'+ extension)
                    
    

def read_bntfile(filepath):
    with open(filepath, 'rb') as fid:
        # Read nrows, ncols, and zmin
        nrows = np.fromfile(fid, dtype=np.uint16, count=1)[0]
        ncols = np.fromfile(fid, dtype=np.uint16, count=1)[0]
        zmin = np.fromfile(fid, dtype=np.float64, count=1)[0]

        # Read the length of the filename
        len_filename = np.fromfile(fid, dtype=np.uint16, count=1)[0]
        
        # Read the filename
        imfile = np.fromfile(fid, dtype=np.uint8, count=len_filename).tobytes().decode('ascii')

        # Read the length of the data
        len_data = np.fromfile(fid, dtype=np.uint32, count=1)[0]
        
        # Read the data and reshape
        data = np.empty((nrows*ncols, 0))
        for _ in range(5):
            data_flat= np.fromfile(fid, dtype=np.float64, count=nrows*ncols).reshape(nrows*ncols,1)
            data = np.hstack((data_flat, data))
         
    return data, zmin, nrows, ncols, imfile

def bnt_to_depth_PNG(path):
    for subject in os.listdir(path): 
        subject_path = os.path.join(path, subject)
        for filename in os.listdir(subject_path):
            if filename.endswith(".bnt"): #only for bnt files
                input_bnt = os.path.join(subject_path, filename)

                data, zmin, nrows, ncols, imfile = read_bntfile(input_bnt)
               
                mask = ~((data[:, 0] == zmin) | (data[:, 1] == zmin) | (data[:, 2] == zmin))
                data_mod = data[mask]
                
                # Create grid for interpolation
                x = np.linspace(np.min(data_mod[:, 0]), np.max(data_mod[:, 0]), ncols)
                y = np.linspace(np.min(data_mod[:, 1]), np.max(data_mod[:, 1]), nrows)
                X, Y = np.meshgrid(x, y)
                
                # Interpolate Z values
                Z = griddata((data_mod[:, 0], data_mod[:, 1]), data_mod[:, 2], (X, Y), method='cubic')
                
                # Adjust Z values for better visualization
                z_max = np.nanmax(Z)
                z_min = np.nanmin(Z)
                val = 200 #200
                if z_min < z_max - val:
                    Z[Z < z_max - val] = z_max - val
                    z_min = z_max - val
                
                #! # Plot the 3D surface
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # X, Y = np.meshgrid(x, y)
                # ax.plot_surface(X, Y, Z, cmap='Grays')
                # plt.show()           
                
                height, width = Z.shape 
                if height > width:
                    new_height = 512
                    new_width = int(512 * (width / height))
                else:
                    new_width = 512
                    new_height = int(512 * (height / width))
                #Resize image 
                Z = np.array(Image.fromarray(Z).resize((new_width, new_height), Image.BICUBIC))

                # Normalize Z to [0, 1] and convert to 16-bit
                Z = (Z - z_min) / (z_max - z_min)
                #invert eaaach pixel value
                Z = 1 - Z
                Z[Z == 1] = 0
                # Convert to 16-bit
                Z = (Z * 65535).astype(np.uint16)
                
                Z = np.rot90(Z, 3) 
                #flip horizontally
                Z = np.fliplr(Z) 

                #pad to 512x512
                pad_height = max(0, 512 - Z.shape[0])
                pad_width = max(0, 512 - Z.shape[1])
                Z_padded = np.pad(Z, ((pad_height // 2, pad_height - pad_height // 2), 
                                              (pad_width // 2, pad_width - pad_width // 2)),
                                  mode='constant', constant_values=0)
                # Crop to 512x512
                Z = Z_padded[:512, :512]
                
                # Convert numpy array to PIL Image
                im = Image.fromarray(Z, mode='I;16')
                
                #! #show gray image
                # plt.imshow(im, cmap='gray')
                # plt.show()
                
                #!load also the rgb image
                try:
                    rgb_image = Image.open(input_bnt.replace('.bnt', '.png'))
                except:
                    print('No RGB image for', input_bnt)
                    continue
                
                #resize to 512x512
                rgb_image = rgb_image.resize((512, 512), Image.BICUBIC)
                #select only pixels where depth is not 0
                rgb_image = np.array(rgb_image)
                mask = Z > 0
                #put 0 values in the rgb image where depth is 0
                rgb_image[~mask] = 0
                rgb_image = Image.fromarray(rgb_image)
                
                # plt.subplot(1, 2, 1)
                # plt.imshow(rgb_image)
                # plt.title('RGB')
                # plt.subplot(1, 2, 2)
                # plt.imshow(im, cmap='gray')
                # plt.title('Depth')
                # plt.show()
                
                depthmap_output = input_bnt.replace('.bnt', '') + '_depthmap.png'
                im.save(depthmap_output)
                rgb_output = input_bnt.replace('.bnt', '') + '_rgb.png'
                rgb_image.save(rgb_output)  
                print('Saved depthmap and rgb image for', input_bnt)
                


#!##
#!MAIN
#!##
if __name__ == '__main__':
    
    #! convert_AUs_to_labels
    #convert_AUs_to_labels('../Datasets/Original_Bosphorus/Subjects')
    
    #!convert every .bnt file in the dataset to .png depthmap
    #bnt_to_depth_PNG('../Datasets/Bosphorus/Subjects')
        
    #! finally delete all file not ending with '_depthmap.png' or '_rgb.png'
    for subject in os.listdir('../Datasets/Bosphorus/Subjects'):
        for filename in os.listdir('../Datasets/Bosphorus/Subjects/' + subject):
            if not filename.endswith('_depthmap.png') and not filename.endswith('_rgb.png'):
                os.remove('../Datasets/Bosphorus/Subjects/' + subject + '/' + filename)
    
    