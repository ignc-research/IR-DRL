import pybullet as pyb
from typing import List
from modular_drl_env.robot.robot_implementations.ur5 import UR5
from ..camera_utils import *
from ..camera import CameraBase # to prevent circular imports the things within the package have to be imported using the relative path

__all__ = [
    'StaticFloatingCameraFollowEffector',
    'StaticFloatingCamera',
]

class StaticFloatingCameraFollowEffector(CameraBase):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, robot : UR5, position: List, target: List = None, camera_args : dict = None, name : str = 'default_floating', **kwargs):
        super().__init__(target= target, camera_args= camera_args, name= name, **kwargs)
        self.robot = robot
        self.pos = position

    def _adapt_to_environment(self):
        self.target = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4]
        super()._adapt_to_environment()

    def get_data_for_logging(self) -> dict:
        """
        Track target because reasons
        """
        dic = super().get_data_for_logging()
        dic[self.output_name + '_target'] = self.target
        return dic


class StaticFloatingCamera(CameraBase):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, position: List, target: List, camera_args : dict = None, name : str = 'default_floating', **kwargs):
        super().__init__(position = position, target= target, camera_args= camera_args, name= name, **kwargs)

    def _adapt_to_environment(self):
        """
        Since there are no changes to the camara's parameters we can just skip updating it
        """
        pass
        # super()._adapt_to_environment()