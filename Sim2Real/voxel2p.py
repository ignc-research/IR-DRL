
# Input: raw depth image by Intel Real Sense (~800k pixels)

# Steps:
# 0. Create a starting voxelization using the old code, virtually subdivide the Pybullet workspace into a 3D boolean grid, manage Pybullet voxel objects via the grid indices of their current worldspace position
# 1. Apply normal depth image to pcd algorithm to input.
#   -> per-pixel boolean mask: does this pixel create a point in the point cloud yes/no? (every nth pixel create a point)
# 2. Take the same boolean mask from the last frame, subtract both ways
#   -> two boolean masks: positive pcd(Changes added), negative pcd (changes to delete)
# 3. Voxelize both pcds
#   -> indices of voxels to delete and to add
# 4. go over all to-delete-voxels and remove pybullet objects from that grid id and put them into reserve
# 5. use the reserve to fill up all to-add-voxels