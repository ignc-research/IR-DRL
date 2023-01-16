from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pybullet as pyb
from world.world import World
from time import time

class Robot(ABC):
    """
    Abstract Base Class for a robot. Methods signed with abstractmethod need to be implemented by subclasses.
    Movement is already implemented and should work if all the class variables are set correctly.
    See the ur5 robot for examples.
    """

    def __init__(self, name: str,
                       world:World,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       end_effector_link_id: int, 
                       base_link_id: int,
                       control_mode: int, 
                       xyz_delta: float,
                       rpy_delta: float,
                       joints_delta: float):
        super().__init__()

        # set name
        self.name = name

        # set id field, this will be given by the world containing this robot
        # it's used by other objects such as goals to access the correct robot's data when it's in some list somewhere
        self.id = None

        # set world
        self.world = world

        # base position
        self.base_position = np.array(base_position)

        # base orientation
        self.base_orientation = np.array(base_orientation)

        # resting pose angles
        self.resting_pose_angles = np.array(resting_angles)

        # use physics sim or simply teleport for movement
        self.use_physics_sim = use_physics_sim

        # link ids
        self.end_effector_link_id = end_effector_link_id
        self.base_link_id = base_link_id

        # PyBullet related variables
        self.object_id = None  # PyBullet object id
        self.joints_ids = []  # array of joint ids, this gets filled at runtime
        self.joints_limits_lower = []  # this and the two below you have to fill for yourself in the subclass in __init__
        self.joints_limits_upper = []  # the values are typically found in the urdf
        self.joints_range = None

        # control mode
        #   0: inverse kinematics
        #   1: joint angles
        #   2: joint velocities
        self.control_mode = control_mode

        # goal associated with the robot
        self.goals = None

        # sensors associated with the robot
        self.sensors = []
        # joint and position sensor (for end effector) are mandatory and thus treated separately
        self.joints_sensor = None
        self.position_rotation_sensor = None

        # maximum deltas on movements, will be used depending on control mode
        self.xyz_delta = xyz_delta
        self.rpy_delta = rpy_delta
        self.joints_delta = joints_delta
        self.joints_vel_delta = None  # this will only be used for joint velocity control mode and must be written to in the build method, see the ur5 implementation for an example
        self.joints_forces = None  # same as the one above

    @abstractmethod
    def get_action_space_dims(self):
        """
        A simple method that should return a tuple containing as first entry the number action space
        dimensions if the joints themselves or their velocities are controlled by the network (this should just be the amount of joints)
        and as second entry the dimensions when running on inverse kinematics (usually 6).
        These numbers get used when constructing the env's action space.
        Put something other than (6,6) if your robot is controlled in some different way, however that means you must
        also overwrite the moveto_*** or action methods below such that they still work.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Method that spawns the robot into the simulation, moves its base to the desired position and orientation
        and sets its joints to the resting angles. Also populates the PyBullet variables with information.
        """
        pass

    def set_joint_sensor(self, joints_sensor):
        """
        Simple setter method for the joint sensor of this robot.
        """
        self.joints_sensor = joints_sensor

    def set_position_rotation_sensor(self, position_rotation_sensor):
        """
        Simple setter for the position and rotation sensor of this robot.
        """
        self.position_rotation_sensor = position_rotation_sensor

    def set_goal(self, goal):
        """
        Simple setter for the goal of this robot.
        """
        self.goal = goal

    def process_action(self, action: np.ndarray):
        """
        This takes an action vector as given as the output of the NN actor and applies it to the robot.
        This vector will always have the size given by get_action_space_dims and will contain values from -1 to 1.
        The method will return its execution time on the cpu.
        """
        cpu_epoch = time()
        if self.control_mode == 0:  # control via inverse kinematics
            pos_delta = action[:3] * self.xyz_delta
            rpy_delta = action[3:] * self.rpy_delta

            new_pos = self.position_rotation_sensor.position + pos_delta
            new_rpy = pyb.getEulerFromQuaternion(self.position_rotation_sensor.rotation.tolist()) + rpy_delta

            self.moveto_xyzrpy(new_pos, new_rpy, self.use_physics_sim)
        elif self.control_mode == 1:  # control via joint angles
            joint_delta = action * self.joints_delta

            new_joints = self.joints_sensor.joints_angles + joint_delta

            self.moveto_joints(new_joints, self.use_physics_sim)
        elif self.control_mode == 2:  # control via joint velocities
            if not self.use_physics_sim:
                raise Exception("Joint velocities control mode only available when using the full physics sim!")
            new_joint_vels = action * self.joints_vel_delta

            self.moveto_joints_vels(new_joint_vels)
        return time() - cpu_epoch

    def moveto_joints_vels(self, desired_joints_velocities: np.ndarray):
        """
        Uses the actual physics simulation to set the joint velocities to desired targets.

        :param desired_joints_velocities: Vector containing the new joint velocities.
        """
        pyb.setJointMotorControlArray(self.object_id, self.joints_ids.tolist(), controlMode=pyb.VELOCITY_CONTROL, targetVelocities=desired_joints_velocities.tolist(), forces=self.joints_forces.tolist())

    def moveto_joints(self, desired_joints_angles: np.ndarray, use_physics_sim: bool):
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
        if use_physics_sim:
            pyb.setJointMotorControlArray(self.object_id, self.joints_ids.tolist(), controlMode=pyb.POSITION_CONTROL, targetPositions=desired_joints_angles.tolist(), forces=self.joints_forces.tolist())
        else:
            for i in range(len(self.joints_ids)):
                pyb.resetJointState(self.object_id, self.joints_ids[i], desired_joints_angles[i])

    def moveto_xyzrpy(self, desired_xyz: np.ndarray, desired_rpy: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position and rpy orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_rpy: Vector containing the desired new rpy orientation of the end effector.
        """
        desired_quat = np.array(pyb.getQuaternionFromEuler(desired_rpy.tolist()))
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints, use_physics_sim)

    def moveto_xyzquat(self, desired_xyz: np.ndarray, desired_quat: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position and quat orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_quat: Vector containing the desired new quaternion orientation of the end effector.
        """
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints, use_physics_sim)

    def moveto_xyz(self, desired_xyz: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position.
        Orientation will not be controlled.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        """
        joints = self._solve_ik(desired_xyz, None)
        self.moveto_joints(joints, use_physics_sim)

    def _solve_ik(self, xyz: np.ndarray, quat:Union[np.ndarray, None]):
        """
        Solves the robot's inverse kinematics for the desired pose.
        Returns the joint angles required

        :param xyz: Vector containing the desired xyz position of the end effector.
        :param quat: Vector containing the desired rotation of the end effector.
        :return: Vector containing the joint angles required to reach the pose.
        """
        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=self.object_id,
            endEffectorLinkIndex=self.end_effector_link_id,
            targetPosition=xyz.tolist(),
            targetOrientation=quat.tolist() if quat is not None else None,
            lowerLimits=self.joints_limits_lower.tolist(),
            upperLimits=self.joints_limits_upper.tolist(),
            jointRanges=self.joints_range.tolist(),
            maxNumIterations=100,
            residualThreshold=.01)
        return np.float32(joints)

    def move_base(self, desired_base_position: np.ndarray, desired_base_orientation: np.ndarray):
        """
        Moves the base of the robot towards the desired position and orientation.

        :param desired_base_position: Vector containing the desired xyz position of the base.
        :param desired_base_orientation: Vector containing the desired rotation of the base.
        """

        self.base_position = desired_base_position
        self.base_orientation = desired_base_orientation
        pyb.resetBasePositionAndOrientation(self.object_id, desired_base_position.tolist(), desired_base_orientation.tolist())