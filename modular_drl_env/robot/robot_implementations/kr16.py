from typing import Union
import numpy as np
import pybullet as pyb
from modular_drl_env.robot.robot import Robot

__all__ = [
    'KR16'
]

class KR16(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta:float=0.005, rpy_delta:float=0.005, joint_vel_mul: float=1):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta, joint_vel_mul)
        # from urdf file
        self.joints_limits_lower = np.array([-3.22885911619, -2.70526034059, -2.26892802759, -6.10865238198, -2.26892802759, -6.10865238198])  
        self.joints_limits_upper = np.array([3.22885911619, 0.610865238198, 2.68780704807, 6.10865238198, 2.26892802759, 6.10865238198])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 5.75958653,  5.75958653, 10.7337749])

        self.end_effector_link_id = 6
        self.base_link_id = 7

        self.urdf_path = "robots/predefined/kr16/urdf/kr16.urdf"

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        self.object_id = self.engine.load_urdf(urdf_path=self.urdf_path, position=self.base_position, orientation=self.base_orientation)
        self.joints_ids = self.engine.get_joints_ids_actuators(self.object_id)

        self.moveto_joints(self.resting_pose_angles, False)     