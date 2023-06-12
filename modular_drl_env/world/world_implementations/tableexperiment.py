from modular_drl_env.world.world import World
import numpy as np
import pybullet as pyb
from modular_drl_env.world.obstacles.human import Human
from modular_drl_env.world.obstacles.shapes import Box
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.urdf_object import URDFObject
import pybullet_data as pyb_d
from random import choice, sample
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'TableExperiment'
]

class TableExperiment(World):
    """
    Implements the table experiment with humans and moving obstacles by Kolja and Kai.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       sim_steps_per_env_step: int,
                       env_id: int,
                       assets_path: str,
                       num_obstacles: int,
                       obstacle_training_schedule: bool=False):
        super().__init__(workspace_boundaries, sim_step, sim_steps_per_env_step, env_id, assets_path)
        # INFO: if multiple robot base positions are given, we will assume that the first one is the main one for the experiment
        # also, we will always assume that the robot base is set up at 0,0,z
        # this will make generating obstacles easier

        self.num_obstacles = num_obstacles

        # table bounds, used for placing obstacles at random
        self.table_bounds_low = [-0.7, -0.7, 1.09]
        self.table_bounds_high = [0.7, 0.7, 1.85]

        # wether num obstacles will be overwritten automatically depending on env success rate, might be useful for training
        self.obstacle_training_schedule = obstacle_training_schedule

        # storage position
        self.position_nowhere = np.array([0, 0, -10])

        # storage variations
        self.mult_pre_gen = 20
        self.active_obstacles = []
        
    def set_up(self):
        # ground plane
        plate = GroundPlate()
        plate.build()
        # table
        table = URDFObject([0, 0, 0], [0, 0, 0, 1], [], self.sim_step, self.sim_steps_per_env_step, 0, pyb_d.getDataPath() + "/table/table.urdf", scale=1.75)
        self.obstacle_objects.append(table)
        table.build()

        for i in range(self.num_obstacles):
            for j in range(self.mult_pre_gen):
                # generate random trajectory
                trajectory = []
                velocity = 0
                if np.random.random() < 0.75:
                    trajectory = [np.array([0, 0, 0])]
                    for _ in range(np.random.randint(low=1, high=4)):
                        direction = np.random.uniform(low=-1, high=1, size=(3,))
                        trajectory_length = np.random.uniform(low=0.05, high=0.25)
                        direction = (trajectory_length / np.linalg.norm(direction)) * direction
                        trajectory.append(direction)
                        velocity = np.random.uniform(low=0.01, high=0.5, size=(1,))
                halfExtents = np.random.uniform(low=0.01, high=0.12, size=(3,)).tolist()
                obst = Box(self.position_nowhere, np.random.normal(size=(4,)), trajectory, self.sim_step, self.sim_steps_per_env_step, velocity, halfExtents)
                obst.build()
                self.obstacle_objects.append(obst)

    def reset(self, success_rate: float):
        
        # reset attributes
        self.ee_starting_points = []
        for obst in self.active_obstacles:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.position_nowhere + offset)
        self.active_obstacles = []

        # dynamically adapt number of obstacles via success rate
        if self.obstacle_training_schedule:
            if success_rate < 0.2:
                obs_mean = 0
            elif success_rate < 0.4:
                obs_mean = 1
            elif success_rate < 0.6:
                obs_mean = 2
            elif success_rate < 0.8:
                obs_mean = 3
            else:
                obs_mean = 5
            self.num_obstacles = round(np.random.normal(loc=obs_mean, scale=1.5))
            self.num_obstacles = min(8, self.num_obstacles)
            self.num_obstacles = max(0, self.num_obstacles)

        # sample obstacles from pre-generated ones and move them into random places
        obst_sample = sample(self.obstacle_objects, self.num_obstacles)
        for obst in obst_sample:
            # generate random position
            position = np.random.uniform(low=self.table_bounds_low, high=self.table_bounds_high, size=(3,))
            obst.move_base(position)
            self.active_obstacles.append(obst)       

        # create valid starting positions and targets
        val = False
        while not val:
            self.position_targets = []
            self.rotation_targets = []
            self.joints_targets = []
            robots_with_starting_points = [robot for robot in self.robots if robot.goal is not None]
            val = self._create_ee_starting_points(robots_with_starting_points, factor=success_rate**3)
            val = val and self._create_position_and_rotation_targets(robots_with_starting_points)
            if val:
                break
            else:
                for obst in self.active_obstacles:
                    position = np.random.uniform(low=self.table_bounds_low, high=self.table_bounds_high, size=(3,))
                    obst.move_base(position)

        # take the last obstacle from the sample and deliberately place it between start and goal
        collision = True
        pos_robot = self.ee_starting_points[0][0]
        pos_goal = self.position_targets[0]
        if len(obst_sample) < 0:
            while collision:
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
                obstacle_pos = dist_obstacle + b * np.random.uniform(0, 0.15) + c * np.random.uniform(0, 0.15)
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
            self.active_obstacles.append(obst_sample[-1])

        # move robot into starting position
        for idx, robot in enumerate(self.robots):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False, robot.controlled_joints_ids)

    def update(self):
        for obstacle in self.active_obstacles:
            obstacle.move_traj()