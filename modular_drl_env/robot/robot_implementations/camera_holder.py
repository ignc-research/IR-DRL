from typing import Union
import numpy as np
import pybullet as pyb
from modular_drl_env.robot.robot import Robot

class CameraHolderUR5_Pybullet(Robot):

    def __init__(self, name: str, world, **kwargs):
        super().__init__(name, world, **kwargs)
        self.joints_limits_lower = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        self.joints_limits_upper = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

        self.end_effector_link_id = 7
        self.base_link_id = 1

        self.joints_max_forces = np.array([300., 300., 300., 300., 300., 300.])
        self.joints_max_velocities = np.array([10., 10., 10., 10., 10., 10.])

        self.urdf_path = "ur5/urdf/ur5.urdf"

    def get_action_space_dims(self):
        return (1,1)  # 6 joints

    def build(self):

        self.object_id = self.engine.load_urdf(urdf_path=self.urdf_path, position=self.base_position, orientation=self.base_orientation)
        self.joints_ids = self.engine.get_joints_ids_actuators(self.object_id)

        self.moveto_joints(self.resting_pose_angles, False) 