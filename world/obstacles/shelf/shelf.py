import pybullet as p
import numpy as np
import random
import os
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
from world.obstacles.urdf_object import URDFObjectGenerated
from ..helpers.urdf_wall_generator import UrdfWallGenerator
from typing import Union
from uuid import uuid4

class ShelfObstacle(URDFObjectGenerated):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, env_id: int, params: dict, scale=1) -> None:
        self.env_id = env_id
        self.params = params 
        self.file_name = self.generate()
        super().__init__(position, rotation, trajectory, move_step, self.file_name, env_id, scale)        

    def generate(self):
        rows = self.params["rows"]
        columns = self.params["cols"]
        element_size = self.params["element_size"]
        shelf_depth = self.params["shelf_depth"]
        wall_thickness = self.params["wall_thickness"]

        xy_offset = element_size / 2
        wall_offset = wall_thickness

        urdf_wall_generator = UrdfWallGenerator()
        for row_idx in range(rows):
            for col_idx in range(columns):
                urdf_wall_generator.add_wall(element_size + wall_offset, wall_thickness, shelf_depth, wall_offset + xy_offset + col_idx * element_size, wall_offset + row_idx * element_size, shelf_depth / 2)
                urdf_wall_generator.add_wall(wall_thickness, element_size + wall_offset, shelf_depth, wall_offset + col_idx * element_size, wall_offset + xy_offset + row_idx * element_size, shelf_depth / 2)

                # closing walls
                if col_idx == columns - 1:
                    urdf_wall_generator.add_wall(wall_thickness, element_size  + wall_offset, shelf_depth, wall_offset + (col_idx + 1) * element_size, wall_offset + xy_offset + row_idx * element_size, shelf_depth / 2)
                if row_idx == rows - 1:
                    urdf_wall_generator.add_wall(element_size + wall_offset, wall_thickness, shelf_depth, wall_offset + xy_offset + col_idx * element_size, wall_offset + (row_idx + 1) * element_size, shelf_depth / 2)
        
        file_name = os.path.join(os.path.dirname(__file__), "shelf_" + str(self.env_id) + ".urdf")

        f = open(file_name, "w")
        f.write(urdf_wall_generator.get_urdf())
        f.close()
        return file_name
        