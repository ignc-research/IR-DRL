from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Tuple, Union, Optional
import numpy as np
import torch

if TYPE_CHECKING:
    from modular_drl_env.robot.robot import Robot
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.sensor import Sensor

class Task(ABC):
    def __init__(self, asset_path:str, headless:bool=True) -> None:
        super().__init__()
        self.asset_path = asset_path  # Path to assets used in simulation
        self.headless = headless  # True if the simulation will not be rendered, otherwise false 

    @abstractmethod
    def set_up(
        self, 
        robots: List[Robot],
        obstacles: List[Obstacle],
        sensors: List[Sensor],
        num_envs: int,
        boundaries: Tuple[float, float, float],
        step_size: float
    ) -> Tuple[List[int], List[int], List [int]]:
        """
        The robot, obstacle and sensor class contains all pramameters about the objects which need to be spawned:
            - Position
            - Rotation
            - Mass
            - Collisions
        Depending on class:
            - (Robot:) Urdf containing load data, mass
            - (Obstacle:)
                - Mass, Colour
                - (Shape:) Cube/Sphere/Cylinder/Mesh and additional parameters

        Sensors are always free floating and not attatched to a robot/obstacle. If a sensor needs to be attatched,
        it is specified in the Robot/Obstacle class.

        After spawning all objects, the set_up function will assign an ID to all objects.
        Each id is unique, even between different classes.

        num_envs: Number of environments which will be simulated in paralles
        boundaries: Maximum amount of space required per environment
        set_size: Amount of time simulated per sim step

        Returns the generated robot, obstacles and sensor ids.
        """
        pass

    @abstractmethod
    def set_joint_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint positions of all robots specified in robot_indices to their respective values specified in positions.
        """
        pass
    
    @abstractmethod
    def set_joint_position_targets(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint position targets of all robots specified in robot_indices to their respective values specified in positions.
        """
        pass

    @abstractmethod
    def set_joint_velocities(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint velocities of all robots specified in robot_indices to their respective values specified in velocities.
        """
        pass
     
    @abstractmethod   
    def set_joint_velocity_targets(
        self,
        velocities: Optional[Union[np.ndarray, torch.Tensor]],
        indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the joint velocities targets of all robots specified in robot_indices to their respective values specified in velocities.
        """
        pass

    @abstractmethod
    def set_local_poses(
        self,
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> None:
        """
        Sets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        pass

    @abstractmethod
    def get_local_poses(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
        """
        pass

    @abstractmethod
    def get_sensor_data(
        self, 
        indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
    ) -> List[List]:
        """
        Gets the sensor data generated by all sensors.
        """
        pass

    @abstractmethod
    def get_collisions(self) -> List[Tuple[int, int]]:
        """
        Returns the ids of objects which are colliding. Updated after each step.
        Example: [(1, 2), (1, 3)] -> Object 1 is colliding with object 2 and 3.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Steps the environment for one timestep
        """
        pass