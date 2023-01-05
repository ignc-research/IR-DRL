from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot

class UR5(Robot):

    def __init__(self, name: str, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], end_effector_link_id: int, base_link_id: int):
        super().__init__(name, base_position, base_orientation, resting_angles, end_effector_link_id, base_link_id)

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        if self.built:
            return
        self.id = pyb.loadURDF("ur5/ur5.urdf", basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True)
        joints_info = [pyb.getJointInfo(self.id, i) for i in range(pyb.getNumJoints(self.id))]
        self.joints_ids = np.array([j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])
        self.joints_limits_lower = np.array([j[8] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])
        self.joints_limits_upper = np.array([j[9] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])

        self.moveto_joints(self.resting_pose_angles)     

        self.built = True