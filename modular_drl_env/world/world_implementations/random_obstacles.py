from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box, Sphere
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
import numpy as np
from random import choice, shuffle, sample
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

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
                       assets_path: str,
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.5, 2],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75],
                       randomize_number_of_obstacles: bool=True,
                       pre_generated_obstacles_mult: int=8
                       ):

        super().__init__(workspace_boundaries, sim_step, env_id, assets_path)

        # num obstacles in each category
        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        # obstacle mesaurements
        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        # obstacle velocities
        self.vel_min, self.vel_max = moving_obstacles_vels

        # trajectory lenghts
        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length

        # bool for whether we will randomize
        self.randomize_number_of_obstacles = randomize_number_of_obstacles

        # the number of pre-generated obstacles per original obstacle
        # e.g. if num_static_obstacles is 3 and this is 5, then we will generate 3*5=15 variations before training starts that will be swapped out at each episode start
        # this avoids costly object spawning during training
        self.pre_generated_obstacles_mult = pre_generated_obstacles_mult

        # location to move geometry that is not needed in episode to
        self.obstacle_storage_location = np.array([0, 0, -10])

        # helper list
        self.active_objects = []
        self.moving_obstacles = []

    def set_up(self):
        # add ground plate
        self.ground_plate = GroundPlate()
        self.ground_plate.build()

        # pre-generate all the obstacles we're going to use
        for i in range(self.num_static_obstacles + self.num_moving_obstacles):
            for _ in range(self.pre_generated_obstacles_mult):
                offset = np.random.uniform(low=-5, high=5, size=(3,))
                if i < self.num_moving_obstacles:
                    # generate a velocity
                    move_step = np.random.uniform(low=self.vel_min, high=self.vel_max) * self.sim_step                   
                    # generate trajectory
                    trajectory = [np.array([0, 0, 0])]
                    for i in range(np.random.randint(low=1, high=4)):
                        direction = np.random.uniform(low=-1, high=1, size=(3,))
                        trajectory_length = np.random.uniform(low=self.trajectory_length_min, high=self.trajectory_length_max)
                        direction = (trajectory_length / np.linalg.norm(direction)) * direction
                        trajectory.append(direction)
                else:
                    move_step = 0
                    trajectory = []
                # box
                if np.random.random() > 0.3:
                    length = np.random.uniform(low=self.box_l_min, high=self.box_l_max)
                    width = np.random.uniform(low=self.box_w_min, high=self.box_w_max)
                    height = np.random.uniform(low=self.box_h_min, high=self.box_h_max)

                    dims = [length, width, height]
                    obst = Box(self.obstacle_storage_location + offset, np.random.normal(size=(4,)), trajectory, move_step, dims)
                # sphere
                else:
                    radius = np.random.uniform(low=self.sphere_r_min, high=self.sphere_r_max)
                    obst = Sphere(self.obstacle_storage_location + offset, trajectory, move_step, radius)
                self.obstacle_objects.append(obst)
                obst.build()

    def reset(self, success_rate: float):
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        # move currently used obstacles into storage
        for obst in self.active_objects[1:]:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.obstacle_storage_location + offset)
        self.active_objects = [self.ground_plate]

        # get number of obstacles for this run
        if self.randomize_number_of_obstacles:
            if self.num_moving_obstacles + self.num_static_obstacles > 0:
                rand_number = choice(range(self.num_moving_obstacles + self.num_static_obstacles)) + 1
            else:
                rand_number = 0
        else:
            rand_number = self.num_moving_obstacles + self.num_static_obstacles

        # sample from pre-generated obstacles and move into position
        obst_sample = sample(self.obstacle_objects, rand_number)
        for obst in obst_sample[:-1]:
            # generate random position
            position = np.random.uniform(low=np.array([self.x_min, self.y_min, self.z_min]), high=np.array([self.x_max, self.y_max, self.z_max]), size=(3,))
            obst.move_base(position)
            self.active_objects.append(obst)

        # generate robot starting positions and targets
        robots_with_starting_points = [robot for robot in self.robots if robot.goal is not None]
        val = False
        while not val:
            for robot in self.robots:
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
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            if not pyb_u.collision:
                val = True
                continue
            self.ee_starting_points = []
        min_dist = min((self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2, (self.z_max - self.z_min) / 2)
        self._create_position_and_rotation_targets(robots_with_starting_points, min_dist=min_dist)

        # move a random object between ee and goal
        collision = True
        pos_robot = self.ee_starting_points[0][0]
        pos_goal = self.position_targets[0]
        if len(obst_sample) > 0:
            tries = 0
            while collision and tries < 200:
                tries += 1
                dist_obstacle = pos_goal + (pos_robot-pos_goal) * np.random.uniform(0.5, 0.75)
                # generate base
                a = (pos_robot-pos_goal) / np.linalg.norm((pos_robot-pos_goal))
                temp_vec = np.random.uniform(low=-1, high=1, size=(3,))
                temp_vec = temp_vec / np.linalg.norm(temp_vec)
                b = np.cross(a, temp_vec)
                b = b / np.linalg.norm(b)
                c = np.cross(a,b)
                c = c / np.linalg.norm(b)
                # set obstacle_pos as linear combi of base without normal_vec
                obstacle_pos = dist_obstacle + b * np.random.uniform(0, 0.075) + c * np.random.uniform(0, 0.075)
                # move obstacle between start and goal pos
                obst_sample[-1].move_base(obstacle_pos)

                # check collision for start
                self.robots[0].moveto_joints(self.ee_starting_points[0][2], False)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                collision = pyb_u.collision
                if pyb_u.collision:
                    continue
                else:
                    # and for goal
                    self.robots[0].moveto_joints(self.joints_targets[0], False, self.robots[0].controlled_joints_ids)
                    pyb_u.perform_collision_check()
                    pyb_u.get_collisions()
                    collision = pyb_u.collision
            if tries < 200:
                self.active_objects.append(obst_sample[-1])
            else:
                obst_sample[-1].move_base(self.obstacle_storage_location)
        
        # move robots to starting position
        for idx, robot in enumerate(self.robots):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

    def update(self):

        for obstacle in self.active_objects:
            obstacle.move_traj()