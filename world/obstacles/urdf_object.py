from world.obstacles.obstacle import Obstacle
import pybullet as pyb
import numpy as np
from typing import Union

class URDFObject(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, urdf_path: str, scale: float=1) -> None:
        super().__init__(position, rotation, trajectory, move_step)

        self.urdf_path = urdf_path
        self.scale = scale

    def build(self) -> int:
        self.object_id = pyb.loadURDF(self.urdf_path, self.position, self.rotation, useFixedBase=True, globalScaling=self.scale)
        return self.object_id