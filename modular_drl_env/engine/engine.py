from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.robot.robot import Robot
    from modular_drl_env.world.obstacles.obstacle import Obstacle

class Engine(ABC):
    """
    Abstract base class that handles calls to physics engine methods in the main environment.py file.
    This does not include specific e.g. sensor or robot implementations, these are handled by their own subclasses.
    """

    def __init__(self, use_physics_sim: bool) -> None:
        super().__init__()
        self.use_physics_sim = use_physics_sim  # determines how objects are moved within the engine, either by brute setting their position or correct physics simulation

    @abstractmethod
    def initialize(self, display_mode: bool, sim_step: float, gravity: list, assets_path: str):
        """
        This method starts the engine in the python code.
        It should also set several attributes using the parameters:
        - display_mode: a bool that determines whether to render a GUI for the user (True) or not (False)
        - sim_step: a float number that determines the sim time that passes with call of Engine.step
        - gravity: a 3-vector that determines the gravitational force along the world xyz-axes
        - assets_path: a string containing the absolute path of the assets folder from where the engine will load in meshes
        """
        pass

    @abstractmethod
    def step(self):
        """
        This method should simply let sim time pass within the engine. This should not apply commands or forces to objects in the simulation on its own.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method should reset the entire simulation, meaning that all objects should be deleted and everything be reset.
        """
        pass

    @abstractmethod
    def perform_collision_check(self, robots: List["Robot"], obstacles: List["Obstacle"]) -> bool:
        """
        Performs a collision check 
        1. between all robots and all obstacles in the world and
        2. between each robot
        """
        pass

    @abstractmethod
    def add_ground_plane(self):
        """
        Adds the default ground plane to the current world
        """
        pass

    @abstractmethod
    def addUserDebugLine(self, lineFromXYZ: List[float], lineToXYZ: List[float]):
        """
        Adds a simple line
        """
        pass
    