from typing import Union
import numpy as np
from modular_drl_env.robot.robot import Robot
from modular_drl_env.util.rrt import bi_rrt
from time import process_time

__all__ = [
    'KukaKr3'
]


class KukaKr3(Robot):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool,
                 base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray],
                 resting_angles: Union[list, np.ndarray], control_mode: int, xyz_delta: float = 0.005,
                 rpy_delta: float = 0.005, joint_vel_mul: float = 1):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation,
                         resting_angles, control_mode, xyz_delta, rpy_delta, joint_vel_mul)
        self.joints_limits_lower = np.deg2rad(np.array([-170, -170, -110, -175, -120, -350]))
        self.joints_limits_upper = np.deg2rad(np.array([170, 50, 155, 175, 120, 350]))
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 10., 10., 10.])

        self.end_effector_link_id = "tool0"
        self.base_link_id = "base_link"

        self.urdf_path = "robots/predefined/kuka_kr3/urdf/kr3r540.urdf"

    def get_action_space_dims(self):
        return (6, 6)  # 6 joints

    def build(self):
        self.object_id = self.engine.load_urdf(urdf_path=self.urdf_path, position=self.base_position,
                                               orientation=self.base_orientation)
        self.joints_ids = self.engine.get_joints_ids_actuators(self.object_id)

        self.moveto_joints(self.resting_pose_angles, False)
