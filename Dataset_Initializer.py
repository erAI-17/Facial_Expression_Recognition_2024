import numpy as np
import cv2
import open3d as o3d
import platform
import matplotlib.pyplot as plt
import os
import math
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

'''
This file allows for training-test split of dataset and data visualization . 
For training-test split different proportion can be chosen (80-20, 90-10)
For data visualization, it shows 1 sample as:
- 2d image, 
- depth map, 
- generate point cloud from 2d+depth_map, 
- generate triangular mesh from 2d+depth map)
'''


#!##
#!GENERAL
#!##
def load_2d_and_3d(path, gender, subjectid, emotion):
    """Loads ALL 2d and corresponding depth map representations for subject, emotion

    Args:
        gender (str): F, M
        subj_id (str):
        emotion (str): anger, surprise,...

    Returns:
        _type_: arrays of 2d images and array of depth maps, ALL for 1 subject
    """
    
    path_images = path + '/' + emotion.capitalize() + '/RGB/'
    path_d_maps = path + '/' + emotion.capitalize() + '/DEPTH/'
    
    images = []
    d_maps = []
    for path in [path_images, path_d_maps]: 
        for filename in os.listdir(path):
            parts = filename.split("_")
            if  (parts[0] == gender and parts[1] == subjectid and parts[3] == emotion):
                
                if (parts[4] == "Color.png"): 
                    #load 2d image
                    image = cv2.imread(path_images + filename)
                    # Convert the image from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    images.append(image)
                    
                if (parts[4] == "Depth.png"): 
                    #load 3d image    
                    d_map = cv2.imread(path_d_maps + filename, cv2.IMREAD_UNCHANGED)
                    d_maps.append(d_map)
                
    return  images, d_maps   
  
def show(image, d_map):
    """Shows a single data sample
    """
    #show image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
    
    #show d_map
    plt.imshow(d_map, cmap='gray')
    plt.colorbar()
    plt.show()
    
    #show pointcloud generated from 2d + depth_map 
    point_cloud = depthmap_to_point_cloud(image, d_map)
      
    ##show mesh generated from 2d + depth_map 
    depthmap_to_mesh(image, d_map)

#!##
#!POINT CLOUD
#!##

def sensor():
    '''
    Computes focal lengths for Intel RealSense3000 used to acquire dataset's images
    '''
    # Convert FoV from degrees to radians
    HFoV_degrees = 73
    VFoV_degrees = 59
    HFoV_radians = HFoV_degrees * (math.pi / 180)
    VFoV_radians = VFoV_degrees * (math.pi / 180)

    # Image resolution
    image_width_px = 640
    image_height_px = 480

    # Calculate focal lengths in pixels
    fx = image_width_px / (2 * math.tan(HFoV_radians / 2))
    fy = image_height_px / (2 * math.tan(VFoV_radians / 2))

    return fx, fy

def depthmap_to_point_cloud(image, depth_map):
    # Camera intrinsics (focal length and principal point)
    fx, fy = sensor() # focal length in x and y direction
    cx = image.shape[1] / 2  # principal point in x direction
    cy = image.shape[0] / 2  # principal point in y direction

    # Create a point cloud from the image and depth map
    points = []
    colors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            z = depth_map[i, j]
            if z > 0:
                x = (i - cx) * z / fx
                y = (j - cy) * z / fy
                points.append([x, y, z])
                colors.append(image[i, j])
    
    point_cloud  = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)  
    
    o3d.visualization.draw_geometries([point_cloud])  
        
    return point_cloud

#!##
#!MESH
#!##

def OLD_depthmap_to_mesh(color_image, d_map):  #?OLD version with also frame pixels with depth=0
    rows, cols = d_map.shape 
    vertices = [] 
    triangles = [] 
    # Normalize the depth map values to 0-255 range 
    d_map = cv2.normalize(d_map, None, 0, 255, cv2.NORM_MINMAX) 
    # Create vertices 
    for i in range(rows): 
        for j in range(cols): 
            z = d_map[i, j] 
            if z!=0: 
                color = color_image[i, j] / 255.0 
                vertices.append([i, j, z, color[0], color[1], color[2]]) 
    # Create triangles 
    for i in range(rows - 1): 
        for j in range(cols - 1): 
            idx = i * cols + j 
            triangles.append([idx, idx + 1, idx + cols]) 
            triangles.append([idx + 1, idx + cols + 1, idx + cols]) 
    # Create mesh 
    mesh = o3d.geometry.TriangleMesh() 
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices)[:, :3])  # XYZ coordinates 
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertices)[:, 3:])  # RGB colors 
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles)) 
    #*visualize 
    o3d.visualization.draw_geometries([mesh]) 
    return mesh 


def depthmap_to_mesh(color_image, d_map):
    rows, cols = d_map.shape
    vertices = []
    triangles = []
    vertex_index_map = {}
    current_index = 0

    # Normalize the depth map values to 0-255 range
    d_map = cv2.normalize(d_map, None, 0, 255, cv2.NORM_MINMAX)
    # Create vertices
    for i in range(rows):
        for j in range(cols):
            z = d_map[i, j]
            if z>0:
                color = color_image[i, j] / 255.0
                vertices.append([i, j, z, color[0], color[1], color[2]])
                vertex_index_map[(i, j)] = current_index
                current_index += 1

    # Create triangles
    for i in range(rows - 1):
        for j in range(cols - 1):
            if (i, j) in vertex_index_map and (i, j + 1) in vertex_index_map and (i + 1, j) in vertex_index_map:
                idx1 = vertex_index_map[(i, j)]
                idx2 = vertex_index_map[(i, j + 1)]
                idx3 = vertex_index_map[(i + 1, j)]
                triangles.append([idx1, idx2, idx3])
            if (i, j + 1) in vertex_index_map and (i + 1, j + 1) in vertex_index_map and (i + 1, j) in vertex_index_map:
                idx1 = vertex_index_map[(i, j + 1)]
                idx2 = vertex_index_map[(i + 1, j + 1)]
                idx3 = vertex_index_map[(i + 1, j)]
                triangles.append([idx1, idx2, idx3])

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices)[:, :3])  # XYZ coordinates
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertices)[:, 3:])  # RGB colors
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    
    # Optionally compute vertex normals for better visualization
    #mesh.compute_vertex_normals()
 
    #visualize
    o3d.visualization.draw_geometries([mesh])
    return mesh


#!##
#!GENERATE ANNOTATION FILES
#!##
def train_test_annotations(test_size):
    """Splits dataset into TrAIN and TEST splits and generate annotation .pkl files where a row represents a sample with this schema:
    subj_id (str): unique code
    code (str): same subj_id for same label, may have multiple samples
    label (str): anger, surprise,...
    add (list(str, str)): list storing additional info ( gender, pose,...) 
    
    """
    #!read all datasets and create unique annotation file where each row has schema [subj_id, code, label, add]
    datasets = ['CalD3r', 'MenD3s']
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'happiness':3, 'neutral':4, 'sadness':5, 'surprise':6}
    for dataset in datasets:
        data = []
        path = '../Datasets/'+ dataset 
        for emotion in emotions.keys(): 
            files_path = path + '/' + emotion.capitalize() + '/RGB' #?analyze only RGB modality because it's enough
            for filename in os.listdir(files_path):  
                subj_id = filename.split("_")[1]   
                code = filename.split("_")[2]
                description_label = filename.split("_")[3]
                add = [filename.split("_")[0]]
                
                new_row = [subj_id, code, description_label, emotions[description_label], add]
                data.append(new_row)
        
        #!split data into train and test dataframes (making sure that all the samples with same subj_id, label and add fall inside the same split)
        grouped = {}
        for sample in data:
            key = (sample[0], sample[2], sample[3], tuple(sample[4]))  # (subj_id, code, label, add)
            grouped.setdefault(key, []).append(sample) #? if key doesn't exist yet in the dictionary, it is set as empty by setdefault()
            
        #?#grouped: {'group1': [[sample1], [sample2],...], 'group2': [[]], ... }    
            
        # Convert grouped dictionary to a list of groups
        groups = list(grouped.values())  
        
        # Shuffle the groups
        np.random.seed(42)
        np.random.shuffle(groups)  
        
        # Split the groups into train and test sets
        train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=42)  #? train_test_split from sklearn, automatically splits the list

        # Flatten the list of groups back into arrays
        train_set = ([sample for group in train_groups for sample in group])
        test_set = ([sample for group in test_groups for sample in group])
        
        #convert to dataframes
        train_df = pd.DataFrame(train_set, columns=['subj_id', 'code', 'description_label', 'label', 'add'])
        test_df  = pd.DataFrame(test_set, columns=['subj_id', 'code', 'description_label', 'label', 'add'])
            
        #save annotation train file
        annotation_file = os.path.join(path, 'annotations_train.pkl')
        with open(annotation_file, 'wb') as file:
            pickle.dump(train_df, file)
            
        #save annotation test file
        annotation_file = os.path.join(path, 'annotations_test.pkl')
        with open(annotation_file, 'wb') as file:
            pickle.dump(test_df, file)
             
    
 
#!##
#!MAIN
#!##
if __name__ == '__main__':
    path = '../Datasets/' + 'MenD3s' #MenD3s   #CalD3r   #C:/Users/studente/Documents/GitHub/Documenti/Github/Datasets/   #../Datasets/
    
    #!#example load of images and depth map for 1 sample
    images, d_maps = load_2d_and_3d(path, gender='F', subjectid='09', emotion='anger') #choose example gender, subj_id and emotion
    ##show
    show(images[0], d_maps[0])
    
    #!generate annotation files for each dataset, TEST and TRAIN
    train_test_annotations(test_size=0.1) #10% test, 90% train
    
    #!check annotation files 
    df = pd.read_pickle(path + '/annotations_train.pkl') 
    df.to_csv('annotations_train.csv', index=False)
    print(df)
    print(df.shape)
    print(df.columns)  
    
    
   
    
        
    
    
    
    
    
    
