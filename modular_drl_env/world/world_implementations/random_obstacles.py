from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box, Sphere
import numpy as np
from random import choice, shuffle

__all__ = [
    'RandomObstacleWorld'
]

class RandomObstacleWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed such that they generally end up between the goal and the starting position of then end effector.
    Depending on the configuration, some of them can be moving in various directions at various speeds.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       env_id: int,
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.5, 2],
                       moving_obstacles_directions: list=[],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75],
                       randomize_number_of_obstacles: bool=True
                       ):
        """
        The world config contains the following parameters:
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param num_static_obstacles: int number that is the amount of static obstacles in the world
        :param num_moving_obstacles: int number that is the amount of moving obstacles in the world
        :param sim_step: float for the time per sim step
        :param box_measurements: List of 6 floats that gives the minimum and maximum dimensions of box shapes in the following order: lmin, lmax, wmin, wmax, hmin, hmax
        :param sphere_measurements: List of 2 floats that gives the minimum and maximum radius of sphere shapes
        :param moving_obstacles_vels: List of 2 floats that gives the minimum and maximum velocity dynamic obstacles can move with
        :param moving_obstacles_directions: List of numpy arrays that contain directions in 3D space among which obstacles can move. If none are given directions are generated in random fashion.
        :param moving_obstacles_trajectory_length: List of 2 floats that contains the minimum and maximum trajectory length of dynamic obstacles.
        :param randomize_number_of_obstacles: Bool that determines if the given number of obstacles is always spawned or just an upper limit for a random choice.
        """
        # TODO: add random rotations for the plates

        super().__init__(workspace_boundaries, sim_step, env_id)

        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        self.vel_min, self.vel_max = moving_obstacles_vels

        self.allowed_directions = [np.array(direction) for direction in moving_obstacles_directions]

        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length

        self.obstacle_objects = []  # list to access the obstacle python objects

        self.randomize_number_of_obstacles = randomize_number_of_obstacles


    def build(self, success_rate: float):
        # add ground plate
        self.objects_ids.append(self.engine.add_ground_plane(np.array([0, 0, -0.01])))

        # determine random number of obstacles, if needed
        if self.randomize_number_of_obstacles:
            if self.num_moving_obstacles + self.num_static_obstacles > 0:
                rand_number = choice(range(self.num_moving_obstacles + self.num_static_obstacles)) + 1
            else:
                rand_number = 0
        else:
            rand_number = self.num_moving_obstacles + self.num_static_obstacles

        # add the obstacles
        for i in range(rand_number):
            # generate a random position in the workspace
            position = np.random.uniform(low=np.array([self.x_min, self.y_min, self.z_min]), high=np.array([self.x_max, self.y_max, self.z_max]), size=(3,))
            
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

        # generate starting points and targets
        robots_with_starting_points = [robot for robot in self.robots_in_world if robot.goal is not None]
        #self._create_ee_starting_points(robots_with_starting_points)
        val = False
        while not val:
            for robot in self.robots_in_world:
                rando = np.random.rand(3)
                x = (self.x_min + self.x_max) / 2 + 0.5 * (rando[0] - 0.5) * (self.x_max - self.x_min)
                y = (self.y_min + self.y_max) / 2 + 0.5 * (rando[1] - 0.5) * (self.y_max - self.y_min)
                z = (self.z_min + self.z_max) / 2 + 0.5 * (rando[2] - 0.5) * (self.z_max - self.z_min)
                standard_rot = np.array([np.pi, 0, np.pi])
                random_rot = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                standard_rot += random_rot * 0.1  
                robot.moveto_xyzrpy(np.array([x,y,z]), standard_rot, False)
                robot.joints_sensor.reset()
                self.ee_starting_points.append((np.array([x,y,z]), standard_rot, robot.joints_sensor.joints_angles))
            self.perform_collision_check()
            if not self.collision:
                val = True
                continue
            self.ee_starting_points = []
        min_dist = min((self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2, (self.z_max - self.z_min) / 2)
        self._create_position_and_rotation_targets(robots_with_starting_points, min_dist=min_dist)

        # move robots to starting position
        for idx, robot in enumerate(self.robots_in_world):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []
        self.aux_object_ids = []

    def update(self):

        for obstacle in self.obstacle_objects:
            obstacle.move()