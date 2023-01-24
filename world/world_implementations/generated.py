from world.world import World
from world.obstacles.pybullet_shapes import Box, Sphere
from world.obstacles.human import Human
import numpy as np
import pybullet as pyb
import yaml
import os
import math
import glob
from random import choice, shuffle
from world.obstacles.obstacle import Obstacle
from world.obstacles.maze.maze import MazeObstacle
from world.obstacles.shelf.shelf import ShelfObstacle
from world.obstacles.urdf_object import URDFObject

__all__ = [
    'GeneratedWorld'
]

URDF_PATH = "./assets/"
def findUrdfs(search_name):
    return list(glob.iglob(os.path.join(URDF_PATH, f"**/{search_name}.urdf"), recursive=True))


def getPosition(obj):
    xyz = obj["position"]
    return [xyz["x"], xyz["y"], xyz["z"]]


def getRotation(obj):
    conversion_fac = math.pi / 180
    rpy = obj["rotation"]
    return np.array(pyb.getQuaternionFromEuler([rpy["r"] * conversion_fac, rpy["p"] * conversion_fac, rpy["y"] * conversion_fac]))


def getTrajectory(obj):
    if "params" not in obj or "move" not in obj["params"]:
        return []
    return list(map(lambda x: np.array(getPosition(x)), obj["params"]["move"]))

def getStep(obj):
    if "params" not in obj or "step" not in obj["params"]:
        return .1
    return obj["params"]["step"]

def getScale(obj):
    if "scale" in obj:
        scale = obj["scale"]
    else:
        scale = 1
    return scale

def load_config(config_path: str):
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)

class GeneratedWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """
    obstacle_objects: list[Obstacle] = []

    def __init__(self, workspace_boundaries: list=[-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], 
                       sim_step: float=1/240 ):
        """
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param sim_step: float for the time per sim step
        """
        super().__init__(workspace_boundaries, sim_step)
        self.config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))


    def load_obstacle(self, obstacle):
        obstacle_name = obstacle["type"]
        position = getPosition(obstacle)
        rotation = getRotation(obstacle)
        scale = getScale(obstacle)
        step = getStep(obstacle)
        trajectory = getTrajectory(obstacle)

        if obstacle_name == "human":
            self.obstacle_objects.append(Human(position, rotation, trajectory, self.sim_step, scale))
        elif obstacle_name == "maze":
            self.obstacle_objects.append(MazeObstacle(position, rotation, trajectory, step, obstacle["params"], scale))
        elif obstacle_name == "shelf":
            self.obstacle_objects.append(ShelfObstacle(position, rotation, trajectory, step, obstacle["params"], scale))
        else:
            urdfs = findUrdfs(obstacle_name)
            if len(urdfs) > 0:
                urdf_name = urdfs[0]
            else:
                urdf_name = f"{urdf_name}.urdf"
            self.obstacle_objects.append(URDFObject(position, rotation, trajectory, step, urdf_name, scale))



    def build(self):
        for obstacle in self.config["obstacles"]:
            self.load_obstacle(obstacle)

        for obstacle in self.obstacle_objects:
            obstacle.build()
            self.objects_ids.append(obstacle.object_id)


    def reset(self):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []

    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move()
        
    def create_ee_starting_points(self):
        self.ee_starting_points.append([None])
        return self.ee_starting_points

    def create_position_target(self):
        self.position_targets.append([0,0,0])
        return self.position_targets

    def create_rotation_target(self) -> list:
        return None  # TODO for later, just adding this so that the world starts

    def build_visual_aux(self):
        pass