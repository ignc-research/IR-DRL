from modular_drl_env.robot.robot import Robot
from ..camera import CameraBase_Pybullet, CameraArgs
from ..camera_utils import directionalVectorsFromQuaternion

from typing import Union, List, Dict

import numpy as np
import pybullet as pyb

__all__ = ['BuddyRobotCamera_Pybullet']

class BuddyRobotCamera_Pybullet(CameraBase_Pybullet):

    def __init__(self, robot_camera : Robot, target : Union[List, Robot], **kwargs):
        self.robot_camera = robot_camera
        position = pyb.getLinkState(self.robot_camera, self.robot_camera.end_effector_link_id)[4]
        self.robot_target = None
        if type(target) is Robot:
            self.robot_target = target
            target = pyb.getLinkState(self.robot_target, self.robot_target.end_effector_link_id)[4]
        elif type(target) is List:
            pass
        else:
            raise ValueError(f'target is of type: {type(target)}, supported types are {Robot} and {List}')

        super().__init__(position= position, target= target, **kwargs)

    def _adapt_to_environment(self):
        self.pos, camera_orientation = pyb.getLinkState(self.robot_camera, self.robot_camera.end_effector_link_id)[4:6]
        self.target = pyb.getLinkState(self.robot_target, self.robot_target.end_effector_link_id)[4]
        self.camera_args['up_vector'] = directionalVectorsFromQuaternion(camera_orientation)[0]
        return super()._adapt_to_environment()
