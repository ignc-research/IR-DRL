from typing import Union
import numpy as np
import pybullet as pyb
from robot.robot import Robot

class UR5(Robot):

    def __init__(self, name: str, world, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], end_effector_link_id: int, base_link_id: int, control_joints:bool, xyz_vel:float, rpy_vel:float, joint_vel:float):
        super().__init__(name, world, base_position, base_orientation, resting_angles, end_effector_link_id, base_link_id, control_joints, xyz_vel, rpy_vel, joint_vel)
        self.joints_limits_lower = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        self.joints_limits_upper = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower

    def get_action_space_dims(self):
        return (6,6)  # 6 joints

    def build(self):

        if self.built:
            return
        self.object_id = pyb.loadURDF("ur5/urdf/ur5.urdf", basePosition=self.base_position.tolist(), baseOrientation=self.base_orientation.tolist(), useFixedBase=True)
        joints_info = [pyb.getJointInfo(self.id, i) for i in range(pyb.getNumJoints(self.id))]
        self.joints_ids = np.array([j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE])

        self.moveto_joints(self.resting_pose_angles)     

        self.built = True