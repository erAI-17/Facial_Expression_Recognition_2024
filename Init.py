import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
depth_maps: 224x224
images: 

'''
gender = 'F' 
subjectid = '001_1120'
emotion = 'anger'

image_path = '../Datasets/CalD3r/Emotions/Anger/Color/'
depth_map_path = '../Datasets/CalD3r/Emotions/Anger/Depth/'
    

def load_image_and_depth_map(gender, subjectid, emotion):
    #load 2d image
    image = cv2.imread(image_path + gender + '_' + subjectid  + '_'+ emotion + '_Color.png')
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #load depth_map
    depth_map = cv2.imread(depth_map_path + gender + '_' +subjectid + '_'+ emotion + '_Depth.png'  , cv2.IMREAD_GRAYSCALE)
    print(depth_map)
    
    return image, depth_map
    
        
def show_depth_map(depth_map):
    # Convert the depth map to a grayscale image
    depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR) 

    # Display the grayscale depth map
    cv2.imshow('Grayscale Depth Map', depth_map_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def normalize(depth_map):
    #depth_map = depth_map.astype(np.float32) * 0.001 
    # Normalize the depth map values to 0-255 range
    depth_map = cv2.normalize(depth_map, None, 0, 600, cv2.NORM_MINMAX)
    
    return depth_map
    
    
def naive_point_cloud_2d_3d(image, depth_map):
    # Camera intrinsics (focal length and principal point)
    fx = 463.8888854980469  # focal length in x direction
    fy = 463.8888854980469  # focal length in y direction
    cx = image.shape[1] / 2  # principal point in x direction
    cy = image.shape[0] / 2  # principal point in y direction

    # Create a point cloud from the image and depth map
    points = []
    colors = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            Z = depth_map[y, x]
            if Z!=0:
                print("diff")
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points.append([X, Y, Z])
            colors.append(image[y, x])
        
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
    
    return point_cloud


def naive_triangle_mesh(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh.compute_vertex_normals()
    
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    #load
    image, depth_map = load_image_and_depth_map(gender, subjectid, emotion)
    
    #normalize depth_map
    #depth_map = normalize(depth_map)
    
    # #!!show
    # plt.imshow(image)
    # plt.axis('off')  # Hide axes
    # plt.show()
    
    show_depth_map(depth_map)
    # #!!
    
    #convert to point cloud
    #point_cloud = naive_point_cloud_2d_3d(image, depth_map)
    
    #point cloud to mesh
    #naive_triangle_mesh(point_cloud)

    
    
    
    
    
