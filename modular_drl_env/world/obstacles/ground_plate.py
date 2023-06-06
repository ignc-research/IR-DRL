from modular_drl_env.world.obstacles.obstacle import Obstacle
import numpy as np
from typing import Union
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class GroundPlate(Obstacle):

    def __init__(self, seen_by_obstacle_sensor: bool=True) -> None:
        super().__init__([0, 0, -0.01], [0, 0, 0, 1], [], 0, 0, 0, seen_by_obstacle_sensor)

    def build(self) -> int:
        self.object_id = pyb_u.add_ground_plane(np.array([0, 0, -0.01]))
        return self.object_id