import pybullet as p
import numpy as np
import random
import os
from typing import Union

from ModEnvDRL.shared.maze_generator import MazeGenerator
from ModEnvDRL.world.obstacles.urdf_object import URDFObjectGenerated

class MazeObstacle(URDFObjectGenerated):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, env_id: int, params: dict, scale=1) -> None:
        self.env_id = env_id
        self.params = params
        self.file_name = self.generate()
        super().__init__(position, rotation, trajectory, move_step, self.file_name, env_id, scale)

    def generate(self):
        generator = MazeGenerator(self.params)

        file_name = os.path.join(os.path.dirname(__file__), "maze_" + str(self.env_id) + ".urdf")

        f = open(file_name, "w")
        f.write(generator.generate())
        f.close()

        self.solution = generator.solution
        return file_name
        