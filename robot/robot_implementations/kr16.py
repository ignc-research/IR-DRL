from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot

__all__ = [
    'KR16'
]

class KR16(Robot):

    def __init__(self, name: str, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta:float, rpy_delta:float):
        super().__init__(name, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta)
        # from urdf file
        self.joints_limits_lower = np.array([-3.22885911619, -2.70526034059, -2.26892802759, -6.10865238198, -2.26892802759, -6.10865238198])  
        self.joints_limits_upper = np.array([3.22885911619, 0.610865238198, 2.68780704807, 6.10865238198, 2.26892802759, 6.10865238198])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.end_effector_link_id = 6
        self.base_link_id = 7

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        self.object_id = pyb.loadURDF("kr16/urdf/kr16.urdf", basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True)
        joints_info = [pyb.getJointInfo(self.id, i) for i in range(pyb.getNumJoints(self.id))]
        self.joints_ids = np.array([j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])
        print(joints_info)

        self.joints_forces = np.array([j[10] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])
        self.joints_vel_delta = np.array([j[11] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])

        self.moveto_joints(self.resting_pose_angles, False)     