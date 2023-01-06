from world.world import World
import numpy as np
import pybullet as pyb

class RandomObstacleWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """

    def __init__(self, workspace_boundaries: list=[-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], 
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.01, 0.35],
                       moving_obstacles_directions: list=[],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75]
                       ):
        """
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param num_static_obstacles: int number that is the amount of static obstacles in the world
        :param num_moving_obstacles: int number that is the amount of moving obstacles in the world
        :param box_measurements: List of 6 floats that gives the minimum and maximum dimensions of box shapes in the following order: lmin, lmax, wmin, wmax, hmin, hmax
        :param sphere_measurements: List of 2 floats that gives the minimum and maximum radius of sphere shapes
        :param moving_obstacles_vels: List of 2 floats that gives the minimum and maximum velocity dynamic obstacles can move with
        :param moving_obstacles_directions: List of numpy arrays that contain directions in 3D space among which obstacles can move. If none are given directions are generated in random fashion.
        :param moving_obstacles_trajectory_length: List of 2 floats that contains the minimum and maximum trajectory length of dynamic obstacles.
        """
        super().__init__(workspace_boundaries)

        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        self.vel_min, self.vel_max = moving_obstacles_vels
        self.vels = None  # to be populated in the build method

        self.allowed_directions = moving_obstacles_directions
        self.directions = None  # to be populated in the build method

        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length
        self.trajectory_lenghts = None # to be populated in the build method

    def build(self):
        
        if self.built:
            return

        # add ground plate
        ground_plate = pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01])
        self.objects_ids.append(ground_plate)

        # add the static obstacles
        for i in range(self.num_static_obstacles):


        self.built = True