from typing import Union, List
import numpy as np
from modular_drl_env.robot.robot import Robot

__all__ = [
    'KR16'
]

class KR16(Robot):

    def __init__(self, name: str,
                       id_num: int,
                       world,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[]):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, ik_xyz_delta, ik_rpy_delta, joint_velocities_overwrite, joint_limits_overwrite, controlled_joints)
        self.end_effector_link_id = "ee_link"
        self.base_link_id = "base_link"

        self.urdf_path = "robots/predefined/kr16/urdf/kr16.urdf"  