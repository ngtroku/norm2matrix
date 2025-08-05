import open3d as o3d
import numpy as np
import json

def grid_partition(points, grid_size): # Divide map into grids

    # initial settings
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    num_iter_x, num_iter_y = int((x_max-x_min) // grid_size + 1), int((y_max-y_min) // grid_size + 1)
    grid_dict = {}

    for xi in range(num_iter_x):
        for yi in range(num_iter_y):
            grid_thres_x, grid_thres_y = x_min + grid_size * xi, y_min + grid_size * yi

            filtered_points = points[(points[:, 0] > grid_thres_x) & (points[:, 0] <= (grid_thres_x + grid_size)) & (points[:, 1] > grid_thres_y) & (points[:, 1] <= (grid_thres_y + grid_size))]

            if (xi, yi) not in grid_dict:
                grid_dict[xi, yi] = []

            grid_dict[(xi, yi)].append(filtered_points)

    return grid_dict, num_iter_x, num_iter_y

def calc_matrix(normal_vector): # calculate transform matrix
    # normalize
    normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    cos_theta = np.dot(normalized_normal_vector, np.array([0, 0, 1]))  
    theta = np.arccos(cos_theta)  

    rotation_axis = np.cross(normalized_normal_vector, np.array([0, 0, 1]))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    ux, uy, uz = rotation_axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    output_matrix = np.array([
    [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta, 0],
    [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta, 0],
    [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta), 0],
    [0, 0, 0, 1]])

    return output_matrix # transform matrix

def plane_estimation(pointcloud): # Plane estimation by RANSAC

    with open("config.json", "r") as f:
        config = json.load(f)

    distance_threshold = config["plane_estimation"]["distance_threshold"]
    ransac_n = config["plane_estimation"]["ransac_n"]
    num_iterations = config["plane_estimation"]["num_iterations"]

    plane_model, inliers = pointcloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations) # param
    return plane_model, inliers

def transform_pointcloud(grid_data, num_iter_x, num_iter_y): # transform pointcloud 

    normal_vector_table = {}
    output_pointcloud = np.empty((0, 3))

    for xi in range(num_iter_x):
        for yi in range(num_iter_y):
            selected_points = grid_data[(xi, yi)]

            # Rotate the part of the plane that was correctly detected.
            if selected_points[0].shape[0] >= 3:

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(selected_points[0])

                orig_centroid_x = np.mean(selected_points[0][:, 0])
                orig_centroid_y = np.mean(selected_points[0][:, 1])
                orig_centroid_z = np.mean(selected_points[0][:, 2])

                # plane estimation
                plane_model, inliers = plane_estimation(pcd)
                [a, b, c, d] = plane_model # [a, b, c] is the normal vector

                # record plane equation
                if (xi, yi) not in normal_vector_table:
                    normal_vector_table[xi, yi] = []
                
                normal_vector_table[(xi, yi)].extend([a, b, c, d, selected_points[0].shape[0]])

                # plane estimation validation
                if abs(c) <= 0.95:
                    raw_points = np.asarray(pcd.points)

                else:
                    # transform
                    trans_matrix = calc_matrix(np.array([a, b, c]))
                    pcd.transform(trans_matrix)
                    raw_points = np.asarray(pcd.points)

                    transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                    inlier_cloud = pcd.select_by_index(inliers)
                    inlier_points = np.asarray(inlier_cloud.points)
                    centroid_x = np.mean(inlier_points[:, 0])
                    centroid_y = np.mean(inlier_points[:, 1])
                    centroid_z = np.mean(inlier_points[:, 2])

                    raw_points[:, 2] -= centroid_z

                    output_pointcloud = np.vstack((output_pointcloud, raw_points))

            else:

                # record plane equation
                if (xi, yi) not in normal_vector_table:
                    normal_vector_table[xi, yi] = []
                
                normal_vector_table[(xi, yi)].extend([0, 0, 0, 0, 0])

    # Rotate areas that could not be detected plane
    for xi in range(num_iter_x):
        for yi in range(num_iter_y):

            selected_points = grid_data[(xi, yi)]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(selected_points[0])

            if selected_points[0].shape[0] >= 3:
                plane_model, inliers = plane_estimation(pcd)
                [a, b, c, d] = plane_model # [a, b, c] is the normal vector

                if abs(c) <= 0.95:
                    normal_vector_table[(xi, yi)][:] = [a, b, c, d, 0]
                    if xi == 0 and yi != 0:
                        weighted_a = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][0] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][0] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][0]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi+1, yi)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][1] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][1] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][1]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi+1, yi)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][2] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][2] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][2]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi+1, yi)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][3] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][3] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][3]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi+1, yi)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z

                        output_pointcloud = np.vstack((output_pointcloud, raw_points))
                    
                    elif xi == num_iter_x-1 and yi != 0:
                        weighted_a = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][0] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][0] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][0]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi-1, yi)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][1] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][1] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][1]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi-1, yi)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][2] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][2] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][2]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi-1, yi)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][3] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][3] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][3]) / (normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + normal_vector_table[(xi-1, yi)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)             

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z       

                    elif xi != 0 and yi == 0:
                        weighted_a = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][0] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][0] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][0]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][1] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][1] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][1]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][2] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][2] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][2]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][3] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][3] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][3]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)                     

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z

                    elif xi != 0 and yi == num_iter_y-1:
                        weighted_a = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][0] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][0] + normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][0]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][1] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][1] + normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][1]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][2] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][2] + normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][2]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][3] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][3] + normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][3]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)   

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z

                    elif xi == 0 and yi == 0:
                        weighted_a = (normal_vector_table[(0, 1)][4] * normal_vector_table[(0, 1)][0] + normal_vector_table[(1, 0)][4] * normal_vector_table[(1, 0)][0]) / (normal_vector_table[(0, 1)][4] + normal_vector_table[(1, 0)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(0, 1)][4] * normal_vector_table[(0, 1)][1] + normal_vector_table[(1, 0)][4] * normal_vector_table[(1, 0)][1]) / (normal_vector_table[(0, 1)][4] + normal_vector_table[(1, 0)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(0, 1)][4] * normal_vector_table[(0, 1)][2] + normal_vector_table[(1, 0)][4] * normal_vector_table[(1, 0)][2]) / (normal_vector_table[(0, 1)][4] + normal_vector_table[(1, 0)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(0, 1)][4] * normal_vector_table[(0, 1)][3] + normal_vector_table[(1, 0)][4] * normal_vector_table[(1, 0)][3]) / (normal_vector_table[(0, 1)][4] + normal_vector_table[(1, 0)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)   

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z

                    else:
                        weighted_a = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][0] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][0] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][0] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][0]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_b = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][1] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][1] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][1] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][1]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_c = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][2] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][2] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][2] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][2]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)
                        weighted_d = (normal_vector_table[(xi, yi-1)][4] * normal_vector_table[(xi, yi-1)][3] + normal_vector_table[(xi-1, yi)][4] * normal_vector_table[(xi-1, yi)][3] + normal_vector_table[(xi, yi+1)][4] * normal_vector_table[(xi, yi+1)][3] + normal_vector_table[(xi+1, yi)][4] * normal_vector_table[(xi+1, yi)][3]) / (normal_vector_table[(xi-1, yi)][4] + normal_vector_table[(xi+1, yi)][4] + normal_vector_table[(xi, yi-1)][4] + normal_vector_table[(xi, yi+1)][4] + 1e-7)

                        trans_matrix = calc_matrix(np.array([weighted_a, weighted_b, weighted_c]))
                        pcd.transform(trans_matrix)
                        raw_points = np.asarray(pcd.points)   

                        transformed_plane_model, transformed_inliers = plane_estimation(pcd)
                        inlier_cloud = pcd.select_by_index(inliers)
                        inlier_points = np.asarray(inlier_cloud.points)
                        centroid_x = np.mean(inlier_points[:, 0])
                        centroid_y = np.mean(inlier_points[:, 1])
                        centroid_z = np.mean(inlier_points[:, 2])

                        raw_points[:, 2] -= centroid_z
                    
                    if abs(weighted_c) <= 0.95:
                        normal_vector_table[(xi, yi)][:] = [weighted_a, weighted_b, weighted_c, weighted_d, 0]
                    else:
                        normal_vector_table[(xi, yi)][:] = [weighted_a, weighted_b, weighted_c, weighted_d, selected_points[0].shape[0]]
                        output_pointcloud = np.vstack((output_pointcloud, raw_points))
            else:
                pass
    
    return output_pointcloud

if __name__ == "__main__":
    print("Process start")
    
    # load config
    with open("config.json", "r") as f:
        config = json.load(f)

    filename = config["io"]["input_file_name"]
    pcd = o3d.io.read_point_cloud(filename)

    if bool(config["preprocess"]["is_preprocess"]):
        voxel_size = float(config["preprocess"]["voxel_size"])
        pcd=pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pass

    points = np.asarray(pcd.points)
    print(f"Input pointcloud is {points.shape[0]} points")

    if len(points) == 0:
        print("Error: Pointcloud data is empty")
        exit()

    grid_size = float(config["grid"]["split_grid_size"])  

    print("Process 1/2 : Grid deviding start")
    grid_data, num_iter_x, num_iter_y = grid_partition(points, grid_size)
    print("Process 1/2 : Grid deviding finished")

    print("Process 2/2 : Pointcloud transformation start")
    output_cloud = transform_pointcloud(grid_data, num_iter_x, num_iter_y)
    print("Process 2/2 : Pointcloud transformation finished")

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(output_cloud)

    if bool(config["postprocess"]["outlier_removal"]):
        nb_neighbors = config["postprocess"]["n_neighbors"]
        std_ratio = config["postprocess"]["std_ratio"]
        cl, ind = new_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    else:
        pass

    o3d.io.write_point_cloud(config["io"]["output_file_name"], new_pcd) # param

    print("Process finished")

