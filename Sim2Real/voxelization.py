# This file tries to outsource some of the voxelization and clustering to be reusable for other files and to ensure a better readibilty in the main file
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import multiprocessing

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