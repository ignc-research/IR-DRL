import numpy as np
import pytorch3d as torch3d
import torch
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures.volumes import Volumes
from pytorch3d.ops.points_to_volumes import add_pointclouds_to_volumes
import pickle
import pybullet as pyb
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from time import sleep, time
import open3d as o3d

x, y, z = 50, 50, 50
size = 0.035
thresh = 1
n = 500
mul = 20

print(x*y*z)

with open('pcd_1.pkl', 'rb') as infile:
    points_orig = pickle.load(infile)
    print(points_orig.shape)
points_orig = np.tile(points_orig, (mul, 1))
colors_orig = np.zeros(points_orig.shape)


start_time = time()
daten_time = 0
erstell_time = 0
voxelier_time = 0
post_time = 0
for i in range(n):
    daten_start = time()
    points = torch.from_numpy(points_orig.astype(np.float32))
    points = points.to('cuda')
    colors = torch.zeros((points.shape[0],1))
    colors = colors.to('cuda')
    daten_time += time() - daten_start

    erstell_start = time()
    pcd = Pointclouds(points=[points], features=[colors])
    erstell_time += time() - erstell_start

    voxelier_start = time()
    volumes = Volumes(densities=torch.zeros(1, 1, x, y, z), features=torch.zeros(1, 1, x, y, z), voxel_size=size)
    volumes = volumes.to('cuda')
    volumes = add_pointclouds_to_volumes(pointclouds=pcd,
                                        initial_volumes=volumes,
                                        mode='nearest')
    voxelier_time += time() - voxelier_start

    post_start = time()
    voxels = volumes.densities()
    voxels = voxels > thresh

    #num_voxels = len(torch.nonzero(voxels))
    voxels = torch.nonzero(voxels).cpu().numpy()
    post_time += time() - post_start

print("Pytorch", (time() - start_time)/n)
print("Daten", daten_time/n)
print('erstellen', erstell_time/n)
print('voxelier_time', voxelier_time/n)
print('post', post_time/n)
voxels = voxels[:, 2:]
print("length:",len(voxels))
#print('Number', num_voxels)

start_time = time()
daten_time = 0
erstell_time = 0
voxelier_time = 0
post_time = 0
for i in range(n):
    points_norms = np.linalg.norm(points_orig, axis=1)
    norm_mask = np.logical_and(points_norms <= 3.0, points_norms > 0.6)
    points_orig = points_orig[norm_mask]  # remove data points that are too far or too close
    colors_orig = colors_orig[norm_mask]
    erstell_start = time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_orig)
    pcd.colors = o3d.utility.Vector3dVector(colors_orig)
    erstell_time += time() - erstell_start

    voxelier_start = time()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)
    voxelier_time += time() - voxelier_start

    post_start = time()
    voxel_data = [(voxel.grid_index, voxel.color) for voxel in voxel_grid.get_voxels()]
    voxels, voxel_colors = zip(*voxel_data)
    voxels = np.array(voxels)
    post_time += time() - post_start

print("Numpy", (time() - start_time)/n)
print("Daten", daten_time/n)
print('erstellen', erstell_time/n)
print('voxelier_time', voxelier_time/n)
print('post', post_time/n)
print("length:", len(voxels))
pyb_u.init('./modular_drl_env/assets/', True, 1/240, [0, 0, 10])
pyb_u.toggle_rendering(False)
pyb_u.add_ground_plane(np.array([0, 0, 0]))

for voxel in voxels:
    pyb_u.create_box(voxel * size, np.array([0, 0, 0, 1]), 0, [size/2, size/2, size/2])

pyb_u.toggle_rendering(True)
sleep(15)