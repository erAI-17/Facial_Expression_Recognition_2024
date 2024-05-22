import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import math

'''
This file allows for data visualization in different flavours. 
Once sample has been selected by specifying 'gender', 'subjectid' and 'emotion', it will show sample as:
- 2d image, 
- depth map, 
- point cloud (from 2d+depth_map), 
- triangular mesh (from point cloud),
- triangular mesh (from depth map) 

As explained in paper point cloud is not continuous leading to a poor mesh.
So, the most promising and used representation will be 2d+depth_map and the mesh obtained from 2d+depth_map.
'''

gender = 'M'
subjectid = '003' 
emotion = 'anger'

path_images = '../Datasets/CalD3r/Emotions/' + emotion.capitalize() + '/Color/'
path_d_map = '../Datasets/CalD3r/Emotions/' + emotion.capitalize() + '/Depth/'
    
def sensor():
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


def show_depth_map(d_map):
    plt.imshow(d_map, cmap='gray')
    plt.colorbar()
    plt.show()


def normalize_depth_map(depth_map):
    # Normalize the depth map values to 0-255 range
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return depth_map


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


def point_cloud_to_mesh(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh.compute_vertex_normals()
    
    #*visualize
    o3d.visualization.draw_geometries([mesh])


def depthmap_to_mesh(color_image, d_map):
    rows, cols = d_map.shape
    vertices = []
    triangles = []

    d_map = normalize_depth_map(d_map)
    # Create vertices
    for i in range(rows):
        for j in range(cols):
            z = d_map[i, j]
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


def load_2d_and_3d(gender, subjectid, emotion):
    images = []
    d_maps = []
    for path in [path_images, path_d_map]: 
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
                    d_map = cv2.imread(path_d_map + filename, cv2.IMREAD_UNCHANGED)
                    d_maps.append(d_map)
                
    return  images, d_maps               
    

if __name__ == '__main__':
 
    #load
    images, d_maps = load_2d_and_3d(gender, subjectid, emotion)
      
    #!show
    plt.imshow(images[0])
    plt.axis('off')  # Hide axes
    plt.show()
    
    show_depth_map(d_maps[0])
    
    #2d+3d to pointcloud
    point_cloud = depthmap_to_point_cloud(images[0], d_maps[0])
    
    #point cloud to mesh
    point_cloud_to_mesh(point_cloud)
    
    #2d+3d to mesh
    depthmap_to_mesh(images[0], d_maps[0])
    
    
    
    
    
