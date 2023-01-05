from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pybullet as pyb

class Robot(ABC):
    """
    Abstract Base Class for a robot. Methods signed with abstractmethod need to be implemented by subclasses.
    Movement is already implemented and should work if all the class variables are set correctly.
    See the ur5 robot for examples.
    """

    def __init__(self, name: str, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], end_effector_link_id: int, base_link_id: int):
        super().__init__()

        # set name
        self.name = name

        # base position
        self.base_position = np.array(base_position)

        # base orientation
        self.base_orientation = np.array(base_orientation)

        # resting pose angles
        self.resting_pose_angles = np.array(resting_angles)

        # link ids
        self.end_effector_link_id = end_effector_link_id
        self.base_link_id = base_link_id

        # PyBullet related variables
        self.id = None  # PyBullet object id
        self.joints_ids = []  # array of joint ids
        self.joints_limits_lower = []
        self.joints_limits_upper = []
        self.joints_range = None

        self.joints_ids = [1,2,3,4,5]
        self.name = "ur5_1"
        self.joints_limits_lower = np.array([0,0,0,0,0])
        self.joints_limits_upper = np.array([np.pi, np.pi, np.pi, np.pi, np.pi])
        self.joints_range = self.joints_limits_upper - self.joints_limits_lower
        self.id = 0

        # set build state
        self.built = False

    @abstractmethod
    def get_action_space_dims(self):
        """
        A simple method that should return a tuple containing as first entry the number action space
        dimensions if the joints themselves are controlled by the network (this should just be the amount of joints)
        and as second entry the dimensions when running on inverse kinematics (usually 6).
        These numbers get used when constructing the env's action space.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Method that spawns the robot into the simulation, moves its base to the desired position and orientation
        and sets its joints to the resting angles. Also populates the PyBullet variables with information.
        Does nothing if self.built is True and must set it to True if it was false.
        """
        pass

    def moveto_joints(self, desired_joints_angles: np.ndarray):
        """
        Moves the robot's joints towards the desired configuration.
        Also automatically clips the input such that no joint limits are violated.

        :param desired_joints_angles: Vector containing the desired new joint angles
        """

        # clip desired angles at max/min
        upper_limit_mask = desired_joints_angles > self.joints_limits_upper
        lower_limit_mask = desired_joints_angles < self.joints_limits_lower
        desired_joints_angles[upper_limit_mask] = self.joints_limits_upper[upper_limit_mask]
        desired_joints_angles[lower_limit_mask] = self.joints_limits_lower[lower_limit_mask]

        # apply movement
        for i in range(len(self.joints_ids)):
            pyb.resetJointState(self.id, self.joints_ids[i], desired_joints_angles[i])

    def moveto_xyzrpy(self, desired_xyz: np.ndarray, desired_rpy: np.ndarray):
        """
        Moves the robot such that end effector is in the desired xyz position and rpy orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_rpy: Vector containing the desired new rpy orientation of the end effector.
        """
        desired_quat = np.array(pyb.getQuaternionFromEuler(desired_rpy.tolist()))
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints)

    def moveto_xyzquat(self, desired_xyz: np.ndarray, desired_quat: np.ndarray):
        """
        Moves the robot such that end effector is in the desired xyz position and quat orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_rpy: Vector containing the desired new quaternion orientation of the end effector.
        """
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints)

    def _solve_ik(self, xyz: np.ndarray, quat:np.ndarray):
        """
        Solves the robot's inverse kinematics for the desired pose.
        Returns the joint angles required

        :param xyz: Vector containing the desired xyz position of the end effector.
        :param quat: Vector containing the desired rotation of the end effector.
        :return: Vector containing the joint angles required to reach the pose.
        """
        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.end_effector_link_id,
            targetPosition=xyz.tolist(),
            targetOrientation=quat.tolist(),
            lowerLimits=self.joints_limits_lower.tolist(),
            upperLimits=self.joints_limits_upper.tolist(),
            jointRanges=self.joints_range.tolist(),
            restPoses=self.resting_pose_angles.tolist(),
            maxNumIterations=2000,
            residualThreshold=5e-3)
        return np.float32(joints)