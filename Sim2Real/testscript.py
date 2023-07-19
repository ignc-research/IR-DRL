import torch
import numpy as np
import pickle
from pytorch3d.ops.points_to_volumes import add_pointclouds_to_volumes
from pytorch3d.structures.volumes import Volumes
from pytorch3d.structures.pointclouds import Pointclouds

# points = np.array([[0.42030528, -0.47338024,  1.7930001,1.],[0.42552346, -0.47443628,  1.79700005,  1.]])

# points_after = np.array([[-0.02030528,0.0569999 ,0.77338024,  1.], [-0.02552346,  0.05299995,  0.77443628,  1.        ]])

# mat = np.array([[-1.00000000e+00, -2.46519033e-32, -1.22464680e-16,  4.00000000e-01],
#  [ 1.22464680e-16, -2.22044605e-16, -1.00000000e+00,  1.85000000e+00],
#  [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16,  3.00000000e-01],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# points_gpu = torch.from_numpy(points)
# points_after_gpu = torch.from_numpy(points_after)
# mat_gpu  = torch.from_numpy(mat)

# print(np.matmul(mat, points.T).T.reshape(-1,4))

# print(torch.matmul(mat_gpu, points_gpu.T).T.reshape(-1,4))

with open('./pcd_' + str(1) + '.pkl', 'rb') as infile:
    points = pickle.load(infile)

points = torch.from_numpy(points.astype(np.float32))

pcd = Pointclouds(points=[points], features=[points])
start_volume = Volumes(densities=torch.zeros(1, 1, 128, 128, 128), voxel_size=0.035)
updated_volume = add_pointclouds_to_volumes(
    pointclouds=pcd.cuda(),
    initial_volumes=start_volume.cuda()
)

print(updated_volume)