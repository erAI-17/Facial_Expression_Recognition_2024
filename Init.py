import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os

'''
depth_maps: 224x224
images: 

'''

gender = 'M' 
subjectid = '018'
emotion = 'disgust'

path_2d = '../Datasets/CalD3r/Emotions/Disgust/Color/'
path_3d = '../Datasets/CalD3r/Emotions/Disgust/Depth/'
    

def load_2d_and_3d(path_2d, gender, subjectid, emotion):
    files_2d = []
    files_3d = []
    for path in [path_2d, path_3d]: 
        for filename in os.listdir(path):
            parts = filename.split("_")
            if  (parts[0] == gender and parts[1] == subjectid and parts[3] == emotion):
                
                if (parts[4] == "Color.png"): 
                    #load 2d image
                    file_2d = cv2.imread(path_2d + filename)
                    # Convert the image from BGR to RGB
                    file_2d = cv2.cvtColor(file_2d, cv2.COLOR_BGR2RGB)
                    
                    files_2d.append(file_2d)
                    
                if (parts[4] == "Depth.png"): 
                    #load 3d image    
                    file_3d = cv2.imread(path_3d + filename, cv2.IMREAD_GRAYSCALE)
                    
                    files_3d.append(file_3d)
                
    return  files_2d, files_3d               
    
def show_depth_map(file_3d):
    # Create an Open3D image from the depth map
    o3d_depth_map = o3d.geometry.Image(file_3d)

    # Visualize the depth map
    o3d.visualization.draw_geometries([o3d_depth_map])
    
    
def normalize(depth_map):
    # Normalize the depth map values to 0-255 range
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map
    
    
def single_depthmap_to_point_cloud(image, depth_map):
    # Camera intrinsics (focal length and principal point)
    fx = 200  # focal length in x direction
    fy = 200  # focal length in y direction
    cx = image.shape[1] / 2  # principal point in x direction
    cy = image.shape[0] / 2  # principal point in y direction

    # Create a point cloud from the image and depth map
    points = []
    colors = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            Z = depth_map[y, x]
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(image[y, x])
    
    point_cloud  = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)    
    
    #visualize depth map 
    show_depth_map(depth_map) 
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    
    return point_cloud

def compute_mean_depth_map(files_3d):
    mean_depth_map = np.zeros_like(files_3d[0], dtype=np.float64)
    num_maps = len(files_3d)

    # Sum up all depth maps
    for file_3d in files_3d:
        mean_depth_map += file_3d.astype(np.float64)

    # Compute mean depth map
    mean_depth_map /= num_maps

    return mean_depth_map

def register_point_clouds(source, target, threshold=0.02):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation

def apply_transformation(point_cloud, transformation):
    return point_cloud.transform(transformation)

def multiple_depth_maps_to_point_cloud(files_2d, files_3d):
    point_clouds = []

    for file_2d, file_3d in zip(files_2d, files_3d):
        file_3d = normalize(file_3d)
        point_cloud = single_depthmap_to_point_cloud(file_2d, file_3d)
        
        point_clouds.append(point_cloud)
    
    # Assume the first point cloud as the reference
    reference_pc = point_clouds[0]  
    merged_cloud = reference_pc
    for point_cloud in point_clouds:
        #transformation = register_point_clouds(point_cloud, reference_pc)
        #aligned_pc = apply_transformation(point_cloud, transformation)
        merged_cloud += point_cloud #aligned_pc

    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([merged_cloud])


def naive_triangle_mesh(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh.compute_vertex_normals()
    
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    #load
    image, depth_map = load_2d_and_3d(path_2d, gender, subjectid, emotion)
      
    # #!!show
    # plt.imshow(image)
    # plt.axis('off')  # Hide axes
    # plt.show()
    
    #show_depth_map(depth_map)
    # #!!
    
    #convert to point cloud
    point_cloud = multiple_depth_maps_to_point_cloud(image, depth_map)
    
    #point cloud to mesh
    #naive_triangle_mesh(point_cloud)

    
    
    
    
    
