from world.world import World
from world.obstacles.pybullet_shapes import Box, Sphere
import numpy as np
import pybullet as pyb
from random import choice, shuffle

__all__ = [
    'RandomObstacleWorld'
]

class RandomObstacleWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """

    def __init__(self, workspace_boundaries: list=[-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], 
                       robot_base_positions: list=[np.array([0.0, -0.12, 0.5])],
                       robot_base_orientations: list=[np.array([0, 0, 0, 1])],
                       sim_step: float=1/240,
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.5, 2],
                       moving_obstacles_directions: list=[],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75]
                       ):
        """
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param num_static_obstacles: int number that is the amount of static obstacles in the world
        :param num_moving_obstacles: int number that is the amount of moving obstacles in the world
        :param sim_step: float for the time per sim step
        :param box_measurements: List of 6 floats that gives the minimum and maximum dimensions of box shapes in the following order: lmin, lmax, wmin, wmax, hmin, hmax
        :param sphere_measurements: List of 2 floats that gives the minimum and maximum radius of sphere shapes
        :param moving_obstacles_vels: List of 2 floats that gives the minimum and maximum velocity dynamic obstacles can move with
        :param moving_obstacles_directions: List of numpy arrays that contain directions in 3D space among which obstacles can move. If none are given directions are generated in random fashion.
        :param moving_obstacles_trajectory_length: List of 2 floats that contains the minimum and maximum trajectory length of dynamic obstacles.
        """
        # TODO: add random rotations for the plates

        super().__init__(workspace_boundaries, robot_base_positions, robot_base_orientations, sim_step)

        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        self.vel_min, self.vel_max = moving_obstacles_vels

        self.allowed_directions = moving_obstacles_directions

        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length

        self.obstacle_objects = []  # list to access the obstacle python objects


    def build(self):

        # add ground plate
        ground_plate = pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01])
        self.objects_ids.append(ground_plate)

        # add the moving obstacles
        for i in range(self.num_moving_obstacles + self.num_static_obstacles):
            # pick a one of the robots' starting positions randomly to...
            idx = choice(range(len(self.ee_starting_points)))
            # ... generate a random position between halfway between it and its target
            position = 0.5*(self.ee_starting_points[idx][0] + self.position_targets[idx] + 0.15*np.random.uniform(low=-1, high=1, size=(3,)))
            
            # moving obstacles
            if i < self.num_moving_obstacles:
                # generate a velocity
                move_step = np.random.uniform(low=self.vel_min, high=self.vel_max) * self.sim_step
                # generate a trajectory length
                trajectory_length = np.random.uniform(low=self.trajectory_length_min, high=self.trajectory_length_max)
                # get the direction from __init__ or, if none are given, generate one at random
                if self.allowed_directions:
                    direction = self.allowed_directions[i]
                else:
                    direction = np.random.uniform(low=-1, high=1, size=(3,))
                direction = (trajectory_length / np.linalg.norm(direction)) * direction
                goal_for_movement = direction + position
                trajectory = [position, goal_for_movement]  # loop between two points
            # static ones
            else:
                move_step = 0
                trajectory = []
                      
            # chance for plates 70%, for spheres 30%
            if np.random.random() > 0.3: 
                # plate
                # generate random size
                length = np.random.uniform(low=self.box_l_min, high=self.box_l_max)
                width = np.random.uniform(low=self.box_w_min, high=self.box_w_max)
                height = np.random.uniform(low=self.box_h_min, high=self.box_h_max)

                # randomly assign lwh to xyz
                dims = [length, width, height]
                shuffle(dims)
                plate = Box(position, [0, 0, 0, 1], trajectory, move_step, dims)
                self.obstacle_objects.append(plate)
                self.objects_ids.append(plate.build())
            else:
                # sphere
                # generate random size
                radius = np.random.uniform(low=self.sphere_r_min, high=self.sphere_r_max)
                sphere = Sphere(position, [0, 0, 0, 1], trajectory, move_step, radius)
                self.obstacle_objects.append(sphere)
                self.objects_ids.append(sphere.build())

    def reset(self):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []
        # the next three don't need to be reset, so commented out
        #self.robots_in_world = []
        #self.robots_with_position = []
        #self.robots_with_orientation = []

    def update(self):

        for obstacle in self.obstacle_objects:
            obstacle.move()
        
    def create_ee_starting_points(self):
        for robot in self.robots_in_world:
            if robot in self.robots_with_position:
                rando = np.random.rand(3)
                x = (self.x_min + self.x_max) / 2 + 0.5 * (rando[0] - 0.5) * (self.x_max - self.x_min)
                y = (self.y_min + self.y_max) / 2 + 0.5 * (rando[1] - 0.5) * (self.y_max - self.y_min)
                z = (self.z_min + self.z_max) / 2 + 0.5 * (rando[2] - 0.5) * (self.z_max - self.z_min)
                standard_rot = np.array([np.pi, 0, np.pi])
                random_rot = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                standard_rot += random_rot * 0.1
                self.ee_starting_points.append((np.array([x, y, z]), np.array(pyb.getQuaternionFromEuler(standard_rot.tolist()))))
            else:
                self.ee_starting_points.append((None, None))
        return self.ee_starting_points

    def create_position_target(self):
        for idx, robot in enumerate(self.robots_in_world):
            if robot in self.robots_with_position:
                while True:
                    rando = np.random.rand(3)
                    x = self.x_min + rando[0] * (self.x_max - self.x_min)
                    y = self.y_min + rando[1] * (self.y_max - self.y_min)
                    z = self.z_min + rando[2] * (self.z_max - self.z_min)
                    target = np.array([x, y, z])
                    if np.linalg.norm(target - self.ee_starting_points[idx][0]) > 0.4:
                        self.position_targets.append(target)
                        break
            else:
                self.position_targets.append([])
        return self.position_targets

    def create_rotation_target(self) -> list:
        return None  # TODO for later, just adding this so that the world starts

    def build_visual_aux(self):
        # create a visual border for the workspace
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])