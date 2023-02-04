from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot

__all__ = [
    'Kukaiiwa'
]

class Kukaiiwa(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta:float, rpy_delta:float):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, xyz_delta, rpy_delta)
        # from urdf file
        self.joints_limits_lower = np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175]))  
        self.joints_limits_upper = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175]))
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.deg2rad(np.array([98., 98., 100., 130., 140. , 180. ,180.]))

        self.end_effector_link_id = 7
        self.base_link_id = 0

    def get_action_space_dims(self):
        return (7,7)  # 7 joints

    def build(self):

        self.object_id = pyb.loadURDF("robots/predefined/kuka_iiwa/model.urdf", basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True)
        joints_info = [pyb.getJointInfo(self.id, i) for i in range(pyb.getNumJoints(self.id))]
        self.joints_ids = np.array([j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])

        self.moveto_joints(self.resting_pose_angles, False)     