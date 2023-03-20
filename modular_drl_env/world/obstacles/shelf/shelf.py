import numpy as np
import os
from modular_drl_env.world.obstacles.urdf_object import URDFObjectGenerated
from typing import Union
from uuid import uuid4
from modular_drl_env.shared.shelf_generator import ShelfGenerator

class ShelfObstacle(URDFObjectGenerated):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, env_id: int, params: dict, scale=[1, 1, 1]) -> None:
        self.env_id = env_id
        self.params = params 
        self.file_name = self.generate()
        super().__init__(position, rotation, trajectory, move_step, self.file_name, env_id, scale)        

    def generate(self):
        generator = ShelfGenerator(self.params)

        file_name = os.path.join(os.path.dirname(__file__), "shelf_" + str(self.env_id) + ".urdf")

        f = open(file_name, "w")
        f.write(generator.generate())
        f.close()
        return file_name
        