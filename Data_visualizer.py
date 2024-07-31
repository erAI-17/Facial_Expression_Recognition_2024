import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
import mediapipe as mp

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

#!##
#!POINT CLOUD
#!##
def depthmap_to_point_cloud(image, depth_map):
    # Camera intrinsics (focal length and principal point)
    fx, fy = 432, 424 # focal length in x and y direction
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
def depthmap_to_mesh(rgb, d_map):
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
                color = rgb[i, j] / 255.0
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
#!LANDMARK EXTRACTION
#!##
def landmark_extraction(img):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    result = face_mesh.process(img)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range(468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * img.shape[1])
            y = int(pt1.y * img.shape[0])
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
    # Display the output
    cv2.imshow('Facial Landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    path = '../Datasets/' + 'CalD3r' #MenD3s   #CalD3r   #C:/Users/studente/Documents/GitHub/Documenti/Github/Datasets/   #../Datasets/
    
    #!#example load of images and depth map for 1 sample
    images, d_maps = load_2d_and_3d(path, gender='M', subjectid='003', emotion='surprise') #choose example gender, subj_id and emotion
    
    #!show 2D and 3D
    #show(images[0], d_maps[0])
    
    #!show pointcloud generated from 2d + depth_map 
    #depthmap_to_point_cloud(images[0], d_maps[0])
      
    #!show mesh generated from 2d + depth_map 
    #depthmap_to_mesh(images[0], d_maps[0])
    
    #!landmark extraction
    landmark_extraction(images[0])