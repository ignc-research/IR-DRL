from modular_drl_env.world.obstacles.obstacle import Obstacle
import numpy as np
from typing import Union
from abc import abstractmethod
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class URDFObject(Obstacle):
    """
    This implements an obstacle that is based off a URDF.
    """

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, urdf_path: str, scale: float=1) -> None:
        super().__init__(position, rotation, trajectory, move_step)

        self.urdf_path = urdf_path
        self.scale = scale

    def build(self) -> int:
        self.object_id = pyb_u.load_urdf(urdf_path=self.urdf_path, position=self.position, orientation=self.orientation, scale=self.scale)
        return self.object_id

class URDFObjectGenerated(URDFObject):
    """
    This is a subclass for URDF objects that generate their own URDF at runtime.
    Has a few adaptions to make parallel environments running at the same time work.
    """

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, urdf_path: str, env_id:int, scale: float=1) -> None:
        super().__init__(position, rotation, trajectory, move_step, urdf_path, scale)
        self.file_name = None  # see below
        self.env_id = env_id  # to prevent parallel file write access

    @abstractmethod
    def generate(self):
        """
        This method generates a URDF file and writes it to disk UNDER A UNIQUE FILENAME (very important to prevent crashes from overlapping file access).
        You can come up with your own way to do this, but I'd suggest using the env_id attribute.
        Ideally, you'd also implement a way such that this method only needs to be called if the attributes relevant for generating change.
        Or use this class in your world class only in that way.
        """
        pass

    