# This file tries to outsource some of the voxelization and clustering to be reusable for other files and to ensure a better readibilty in the main file
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import multiprocessing

import open3d as o3d
from sklearn.neighbors import NearestNeighbors

### SOR (Stastical Outlier Removal) (Not so resource intensiv and gets rid of most the free floating voxels (not all)) ########


def statistical_outlier_removal(pcd, n_neighbors=20, std_ratio=2.0):
    points = np.asarray(pcd.points)  # Convert the points to a numpy array

    # Calculate the distances and indices using sklearn's NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(points)
    distances, indices = neigh.kneighbors(points)

    # Calculate mean and standard deviation
    mean_distances = np.mean(distances, axis=1)
    std_distances = np.std(distances, axis=1)

    # Identify points with a distance larger than mean + std_ratio * std_deviation
    outlier_mask = mean_distances > mean_distances.mean() + std_ratio * mean_distances.std()

    # Remove the outliers
    filtered_points = points[~outlier_mask]

    # Create a new PointCloud object for the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_colors = colors[~outlier_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd


#new kdtree 
"""
def get_neighbouring_voxels_idx(kd_tree, current_voxel, neighbourhood_threshold):
    return kd_tree.query_ball_point(current_voxel, neighbourhood_threshold)

def get_voxel_cluster(voxel_centers, neighbourhood_threshold):
    # Build k-d tree for efficient distance calculation
    kd_tree = cKDTree(voxel_centers)
    
    # Use k-d tree to find neighboring voxels for each voxel
    res = [get_neighbouring_voxels_idx(kd_tree, voxel, neighbourhood_threshold) for voxel in voxel_centers]

    # Cluster voxels
    voxel_cluster = np.repeat(-1, voxel_centers.shape[0])
    max_cluster_num = 0
    for i in range(len(voxel_centers)):
        if voxel_cluster[i] < 0:
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                if voxel_cluster[idx] < 0:
                    voxel_cluster[idx] = max_cluster_num
                    queue.extend(res[idx])
            max_cluster_num += 1
    return voxel_cluster
"""


############# KD- TREE CLUSTERING (Very Resource intensive, but eliminates all free floating voxels) ############################
#def neighbors_in_bubble(voxel, voxel_centers, neighbourhood_threshold):
def neighbors_in_bubble(voxel):
    voxel_centers = np.frombuffer(voxel_centers_mp).reshape(voxel_shape_mp)
    norms = np.linalg.norm(voxel - voxel_centers, axis=1)
    #print(norms[:10])
    #print((norms < neighbourhood_threshold)[:10])
    #print(np.where(norms < neighbourhood_threshold)[0][:10])
    return np.where(norms < neighbourhood_threshold_mp)[0]

def get_voxel_cluster(voxel_centers, neighbourhood_threshold):
    # Build k-d tree for efficient distance calculation
    #kd_tree = cKDTree(voxel_centers)

    voxel_centers_mp_buffer = multiprocessing.RawArray('d', voxel_centers.shape[0] * voxel_centers.shape[1])
    #numpy wrapper: 
    voxel_centers_mp_buffer_np = np.frombuffer(voxel_centers_mp_buffer, dtype=np.float64).reshape(voxel_centers.shape)
    np.copyto(voxel_centers_mp_buffer_np, voxel_centers)

    def init_worker(voxel_centers_mp_buffer, voxel_shape, neighbourhood_threshold):
        global voxel_centers_mp
        global voxel_shape_mp
        global neighbourhood_threshold_mp
        voxel_centers_mp = voxel_centers_mp_buffer
        voxel_shape_mp = voxel_shape
        neighbourhood_threshold_mp = neighbourhood_threshold
#Works best with 4 atm
    with multiprocessing.Pool(4, initializer=init_worker, initargs=(voxel_centers_mp_buffer, voxel_centers.shape, neighbourhood_threshold)) as p:
        res = p.map(neighbors_in_bubble, voxel_centers)
    

    voxel_cluster = np.repeat(-1, voxel_centers.shape[0])
    max_cluster_num = 0
    for i in range(len(voxel_centers)):
        if voxel_cluster[i] < 0:
            #set_clusters(i, kd_tree, voxel_centers, voxel_cluster, max_cluster_num, neighbourhood_threshold)
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                #current_voxel = voxel_centers[idx]
                if voxel_cluster[idx] < 0:
                    voxel_cluster[idx] = max_cluster_num
                    #neighbors = get_neighbouring_voxels_idx(kd_tree, current_voxel, neighbourhood_threshold)
                    queue.extend(res[idx])
            max_cluster_num += 1
    return voxel_cluster





def set_clusters(initial_voxel_idx, kd_tree, voxel_centers, voxel_cluster, cluster_num, neighbourhood_threshold):
    queue = deque([initial_voxel_idx])
    while queue:
        idx = queue.popleft()
        current_voxel = voxel_centers[idx]
        if voxel_cluster[idx] < 0:
            voxel_cluster[idx] = cluster_num
            neighbors = get_neighbouring_voxels_idx(kd_tree, current_voxel, neighbourhood_threshold)
            queue.extend(neighbors)

def get_neighbouring_voxels_idx(kd_tree, voxel, neighbour_threshold):
    _, voxels_in_cluster_idx = kd_tree.query(voxel, k=len(kd_tree.data), distance_upper_bound=neighbour_threshold)    
    valid_neighbors = [idx for idx in voxels_in_cluster_idx if idx != kd_tree.n] # Exclude out-of-range indices
    return valid_neighbors

#Experimental mix approach with KD-tree Clustering and Multiprocessing
"""
# This will run in a separate process
def parallel_kdtree_query(voxel_chunk):
    # Each process will have its own k-d tree built from its chunk of voxel_centers
    kd_tree = cKDTree(voxel_chunk)
    return [kd_tree.query_ball_point(voxel, neighbourhood_threshold_mp) for voxel in voxel_chunk]

def get_voxel_cluster(voxel_centers, neighbourhood_threshold):
    global neighbourhood_threshold_mp
    neighbourhood_threshold_mp = neighbourhood_threshold
    
    num_processes = 4
    # Split voxel_centers into chunks for each process
    chunks = np.array_split(voxel_centers, num_processes)
    
    # Use a pool of processes to compute the neighboring voxels for each chunk of voxel_centers
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(parallel_kdtree_query, chunks)
        
    # Flatten the results to get a single list
    res = [item for sublist in results for item in sublist]

    # The rest of the clustering remains the same
    voxel_cluster = np.repeat(-1, voxel_centers.shape[0])
    max_cluster_num = 0
    for i in range(len(voxel_centers)):
        if voxel_cluster[i] < 0:
            queue = deque([i])
            while queue:
                idx = queue.popleft()
                if voxel_cluster[idx] < 0:
                    voxel_cluster[idx] = max_cluster_num
                    queue.extend(res[idx])
            max_cluster_num += 1
    return voxel_cluster
"""