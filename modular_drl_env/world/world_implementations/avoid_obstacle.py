import numpy as np

from modular_drl_env.world.obstacles.shapes import Sphere
from modular_drl_env.world.world import World
import pybullet_data as pyb_d
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'AvoidObstacle'
]


class AvoidObstacle(World):
    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int,
                 moving_obstacles_vels=None):
        super().__init__(workspace_boundaries, sim_step, env_id)

        if moving_obstacles_vels is None:
            moving_obstacles_vels = [0.5, 2]
        self.vel_min, self.vel_max = moving_obstacles_vels

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        for obj in self.obstacle_objects:
            del obj
        self.obstacle_objects = []
        self.aux_object_ids = []

    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.trajectory = [obstacle.trajectory[0], self.robots[0].position_rotation_sensor.position]
            obstacle.move()

    def build(self, success_rate: float):
        # add ground plate
        self.objects_ids.append(pyb_u.add_ground_plane(np.array([0, 0, -0.01])))

        # table
        self.objects_ids.append(pyb_u.load_urdf(pyb_d.getDataPath() + "/table/table.urdf", np.array([0, 0, 0]), np.array([0, 0, 0, 1]), 1.75))

        # generate starting points and targets
        robots_with_starting_points = [robot for robot in self.robots if robot.goal is not None]
        self._create_ee_starting_points(robots_with_starting_points)
        min_dist = min((self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2, (self.z_max - self.z_min) / 2)
        self._create_position_and_rotation_targets(robots_with_starting_points, min_dist=min_dist)

        # move robots to starting position
        for idx, robot in enumerate(self.robots):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

        # generate a random position in the workspace
        position = np.random.uniform(low=np.array([self.x_min, self.y_min, self.z_min]),
                                     high=np.array([self.x_max, self.y_max, self.z_max]), size=(3,))
        # generate a velocity
        move_step = np.random.uniform(low=self.vel_min, high=self.vel_max) * self.sim_step
        # generate a trajectory length
        # get the direction from __init__ or, if none are given, generate one at random
        trajectory = [position, self.ee_starting_points[0][0]]  # loop between two points
        # sphere
        # generate random size
        radius = 0.05
        sphere = Sphere(position, [0, 0, 0, 1], trajectory, move_step, radius)
        self.obstacle_objects.append(sphere)
        self.objects_ids.append(sphere.build())
