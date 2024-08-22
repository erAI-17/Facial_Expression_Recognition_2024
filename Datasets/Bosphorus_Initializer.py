import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from PIL import Image, ImageOps
import cv2
import PIL.Image as Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from skimage.transform import resize


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
        for col in range(5):
            data_flat= np.fromfile(fid, dtype=np.float64, count=nrows*ncols).reshape(nrows*ncols,1)
            data = np.hstack((data_flat, data))
         
    return data, zmin, nrows, ncols, imfile

def bnt_to_depth_PNG(files_path):
    count = 0
    for subject in os.listdir(files_path): 
        subject_path = os.path.join(files_path, subject)
        for filename in os.listdir(subject_path):
            if filename.endswith(".bnt") and filename.split('_')[1] == 'E': #only for emotions
                input_bnt = os.path.join(subject_path, filename)
                output_png = subject_path

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
                if z_min < z_max - 200:
                    Z[Z < z_max - 200] = z_max - 200
                    z_min = z_max - 200
                
                #! # Plot the 3D surface
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # X, Y = np.meshgrid(x, y)
                # ax.plot_surface(X, Y, Z, cmap='Grays')
                # plt.show()           
                
                # Normalize Z to [0, 1] and convert to 16-bit
                Z = (Z - z_min) / (z_max - z_min)
                Z = np.rot90(Z, 3)  

                # Convert to 16-bit
                Z = (Z * 65535).astype(np.uint16)

                # Resize to have the largest dimension as 224 while maintaining aspect ratio
                height, width = Z.shape
                if height > width:
                    new_height = 224
                    new_width = int(224 * (width / height))
                else:
                    new_width = 224
                    new_height = int(224 * (height / width))
                
                # Resize image without interpolation
                Z_resized = np.array(Image.fromarray(Z).resize((new_width, new_height), Image.NEAREST))

                #pad to 224,224
                pad_height = max(0, 224 - Z_resized.shape[0])
                pad_width = max(0, 224 - Z_resized.shape[1])
                Z_padded = np.pad(Z_resized, ((pad_height // 2, pad_height - pad_height // 2), 
                                              (pad_width // 2, pad_width - pad_width // 2)),
                                  mode='constant', constant_values=0)
                
                # Crop to 224x224
                Z_final = Z_padded[:224, :224]
                
                # Convert numpy array to PIL Image
                im = Image.fromarray(Z_final, mode='I;16')
                
                #! #show gray image
                # plt.imshow(im, cmap='gray')
                # plt.show()
                
                count = count + 1
                imfile = imfile.split('.')[0] + '_depthmap.png'
                save_path = os.path.join(output_png, imfile)
                im.save(save_path)
                
                print('counter:', count)
                

#!##
#!MAIN
#!##
if __name__ == '__main__':
    
    #!convert every .bnt file in the dataset to .png depthmap
    files_path = '../Datasets/Bosphorus/Data'
    bnt_to_depth_PNG(files_path)
    
    #!generate annotation files for each dataset, TEST and TRAIN
    # class_distribution, mean, std = train_test_annotations(test_size=0.2) #20% test, 80% train
    
    # #plot histogram class distribution
    # class_distribution = class_distribution['Color']
    # plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue', alpha=0.8)
    # #plt.xlabel('Class')
    # #plt.ylabel('Frequency')
    # #plt.title('Distribution of Classes')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--', linewidth=0.5)
    # plt.tight_layout()  # Adjust layout for better spacing
    # plt.show()
    
    # #!check annotation files 
    # df = pd.read_pickle('../Datasets/' + '/annotations_test.pkl') 
    # #df.to_csv('annotations_train.csv', index=False)
    # print(df)
    # print(df.shape)
    # print(df.columns)  
    