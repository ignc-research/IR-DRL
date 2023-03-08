from typing import Union
import numpy as np
import pybullet as pyb
from modular_drl_env.robot.robot import Robot

__all__ = [
    'Kukaiiwa'
]


class Kukaiiwa(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool,
                 base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray],
                 resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta: float, rpy_delta: float):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation,
                         resting_angles, control_mode, xyz_delta, rpy_delta)
        # from urdf file
        self.joints_limits_lower = np.deg2rad(np.array([-170, -120, -170, -120, -170, -120, -175]))
        self.joints_limits_upper = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175]))
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.deg2rad(np.array([98., 98., 100., 130., 140., 180., 180.]))

        self.end_effector_link_id = 6
        self.base_link_id = 0

        self.urdf_path = "robots/predefined/kuka_iiwa/model.urdf"

    def get_action_space_dims(self):
        return (7, 6)  # 7 joints

    def build(self):
        self.object_id = self.engine.load_urdf(urdf_path=self.urdf_path, position=self.base_position, orientation=self.base_orientation)
        self.joints_ids = self.engine.get_joints_ids_actuators(self.object_id)

        self.moveto_joints(self.resting_pose_angles, False)
