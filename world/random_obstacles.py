from world.world import World
import numpy as np
import pybullet as pyb
from random import choice, shuffle

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
            for idx in choice(range(len(self.robots_with_position))):
                position = 0.5*(self.ee_starting_points[idx] + self.position_targets[idx] + 0.05*np.random.uniform(low=-1, high=1, size=(3,)))
                # chance for plates 70%, for speres 30%
                if np.random() > 0.3: 
                    # plate
                    plate = self._create_plate(position)
                    self.objects_ids.append(plate)
                else:
                    # sphere
                    sphere = self._create_sphere(position)
                    self.objects_ids.append(sphere)

        self.built = True

    def _create_plate(self, position):
        length = np.random.uniform(low=self.box_l_min, high=self.box_l_max)
        width = np.random.uniform(low=self.box_w_min, high=self.box_w_max)
        height = np.random.uniform(low=self.box_h_min, high=self.box_h_max)

        # randomly assign lwh to xyz
        dims = [length, width, height]
        random_dim = shuffle([0, 1, 2])

        plate = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[dims[random_dim[0]], dims[random_dim[1]], dims[random_dim[2]]], rgbaColor=[0, 0, 1, 1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[dims[random_dim[0]], dims[random_dim[1]], dims[random_dim[2]]]),
                                    basePosition=position)
        return plate



    def _create_sphere(self, position):
        radius = np.random.uniform(low=self.sphere_r_min, high=self.sphere_r_max)
        sphere = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=radius, rgba_color=[1, 0, 0, 1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, radius=radius),
                                    basePosition=position)
        return sphere
        

    def _create_ee_starting_points(self) -> list:
        for robot in self.robots_with_position:
            rando = np.random.rand(3)
            x = (self.x_min + self.x_max) / 2 + 0.5 * (rando[0] - 0.5) * (self.x_max - self.x_min)
            y = (self.y_min + self.y_max) / 2 + 0.5 * (rando[1] - 0.5) * (self.y_max - self.y_min)
            z = (self.z_min + self.z_max) / 2 + 0.5 * (rando[2] - 0.5) * (self.z_max - self.z_min)
            self.ee_starting_points.append(np.array([x, y, z]))

    def _create_position_target(self) -> list:
        for idx, robot in enumerate(self.robot_base_positions):
            while True:
                rando = np.random.rand(3)
                x = self.x_min + 0.5 * rando[0] * (self.x_max - self.x_min)
                y = self.y_min + 0.5 * rando[1] * (self.y_max - self.y_min)
                z = self.z_min + 0.5 * rando[2] * (self.z_max - self.z_min)
                target = np.array([x, y, z])
                if np.linalg.norm(target - self.ee_starting_points[idx]) > 0.4:
                    self.position_targets.append(target)
                    break