from modular_drl_env.world.world import World
import numpy as np
import pybullet as pyb
from modular_drl_env.world.obstacles.human import Human
from modular_drl_env.world.obstacles.shapes import Box
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
                       env_id: int,
                       assets_path: str,
                       num_obstacles: int,
                       obstacle_training_schedule: bool=False):
        super().__init__(workspace_boundaries, sim_step, env_id, assets_path)
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
        self.mult_pre_gen = 40
        self.active_obstacles = []
        
    def set_up(self):
        # ground plane
        pyb_u.add_ground_plane(np.array([0, 0, -0.01]))
        # table
        pyb_u.load_urdf(pyb_d.getDataPath() + "/table/table.urdf", np.array([0, 0, 0]), np.array([0, 0, 0, 1]), scale=1.75)

        for i in range(self.num_obstacles):
            for j in range(self.mult_pre_gen):
                # generate random trajectory
                trajectory = []
                move_step = 0
                if np.random.random() < 0.75:
                    trajectory = [np.array([0, 0, 0])]
                    for _ in range(np.random.randint(low=1, high=4)):
                        direction = np.random.uniform(low=-1, high=1, size=(3,))
                        trajectory_length = np.random.uniform(low=0.05, high=0.25)
                        direction = (trajectory_length / np.linalg.norm(direction)) * direction
                        trajectory.append(direction)
                        move_step = np.random.uniform(low=0.01, high=0.5, size=(1,)) * self.sim_step
                halfExtents = np.random.uniform(low=0.01, high=0.12, size=(3,)).tolist()
                obst = Box(self.position_nowhere, [0, 0, 0, 1], trajectory, move_step, halfExtents)
                obst.build()
                self.obstacle_objects.append(obst)

    def reset(self, success_rate: float):
        
        self.ee_starting_points = []
        for obst in self.active_obstacles:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.position_nowhere + offset)
        self.active_obstacles = []

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

        for obst in sample(self.obstacle_objects, self.num_obstacles):
            # generate random position
            position = np.random.uniform(low=self.table_bounds_low, high=self.table_bounds_high, size=(3,))
            obst.move_base(position)
            self.active_obstacles.append(obst)

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

        for idx, robot in enumerate(self.robots):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

    def update(self):
        for obstacle in self.active_obstacles:
            obstacle.move_traj()