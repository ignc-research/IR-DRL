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

    def __init__(self, robot_config):
        super().__init__()

        # set name
        self.name = robot_config["name"]

        # set id field, this will be given by the world containing this robot
        # it's used by other objects such as goals to access the correct robot's data when it's in some list somewhere
        self.id = robot_config["id"]

        # set world
        self.world = robot_config["world"]

        # set sim step
        self.sim_step = robot_config["sim_step"]

        # base position
        self.base_position = np.array(robot_config["base_position"])

        # base orientation
        self.base_orientation = np.array(robot_config["base_orientation"])

        # resting pose angles
        self.resting_pose_angles = np.array(robot_config["resting_angles"])

        # use physics sim or simply teleport for movement
        self.use_physics_sim = robot_config["use_physics_sim"]

        # link ids, these have to be set in your subclass!
        self.end_effector_link_id = None
        self.base_link_id = None

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
        self.control_mode = robot_config["control_mode"]

        # goal associated with the robot
        self.goal = None

        # sensors associated with the robot
        self.sensors = []
        # joint and position sensor (for end effector) are mandatory and thus treated separately
        self.joints_sensor = None
        self.position_rotation_sensor = None

        # maximum deltas on movements, will be used in Inverse Kinematics control
        self.xyz_delta = robot_config["xyz_delta"]
        self.rpy_delta = robot_config["rpy_delta"]

        # velocity and force attributes
        self.joints_vel_delta = None  # must be written to in the build method after loading the URDF with a PyBullet call, see the ur5 implementation for an example
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
        # TODO: deal with joints with two or more degrees of freedom
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
        if self.control_mode == 0:  
            # control via inverse kinematics:
            # actions are small changes in xyz and rpy of the robot's end effector
            # we calculate the changed position, then use inverse kinematics to get the equivalent joint angles
            # then we apply those
            pos_delta = action[:3] * self.xyz_delta
            rpy_delta = action[3:] * self.rpy_delta

            new_pos = self.position_rotation_sensor.position + pos_delta
            new_rpy = pyb.getEulerFromQuaternion(self.position_rotation_sensor.rotation.tolist()) + rpy_delta

            self.moveto_xyzrpy(new_pos, new_rpy, self.use_physics_sim)
        elif self.control_mode == 1:  
            # control via joint angles
            # actions are the new desired joint angles themselves
            # we apply them mostly as is

            # transform action (-1 to 1) to desired new joint angles
            new_joints = action * (self.joints_range / 2) + (self.joints_limits_lower + self.joints_limits_upper) / 2
            
            # if we don't use the physics sim, which will only perform a step towards the desired new joints, 
            # we have to clamp the new joint angles such that they move with at most the maximum velocity within the next sim step
            if not self.use_physics_sim:
                # compute the maximum step we do in that direction
                joint_delta = new_joints - self.joints_sensor.joints_angles
                joint_delta = joint_delta / np.linalg.norm(joint_delta)
                joint_delta = joint_delta * self.joints_vel_delta * self.sim_step

                # compute the joint angles we can actually go to
                new_joints = joint_delta + self.joints_sensor.joints_angles

            # execute movement
            self.moveto_joints(new_joints, self.use_physics_sim)

        elif self.control_mode == 2:  
            # control via joint velocities
            # actions are joint velocities
            # if we use the physics sim, PyBullet can deal with those on its own
            # if we don't, we run simple algebra to get the new joint angles for this step and then apply them

            # transform action (-1 to 1) to joint velocities
            new_joint_vels = action * self.joints_vel_delta
            if not self.use_physics_sim:
                # compute the delta for this sim step
                joint_delta = new_joint_vels * self.sim_step
                # add the delta to current joint angles
                new_joints = joint_delta + self.joints_sensor.joints_angles
                # execute movement
                self.moveto_joints(new_joints, False)

            else:
                # use PyBullet to apply these velocities to robot
                self.moveto_joints_vels(new_joint_vels)
        elif self.control_mode == 3:
            # control robot via relative deviation from trajectory point
            delta_upper = np.asarray(self.joints_limits_upper) - self.trajectory_point
            delta_lower = np.asarray(self.joints_limits_lower) - self.trajectory_point
            deltas = np.where(action > 0, delta_upper[:5], delta_lower[:5])
            new_joints = self.trajectory_point[:5] + deltas[:5] * np.abs(action)
            new_joints = np.append(new_joints, self.joints_sensor.joints_angles[-1])

            # compute the maximum step we do in that direction
            joint_delta = new_joints - self.joints_sensor.joints_angles
            joint_delta = joint_delta / np.linalg.norm(joint_delta)
            joint_delta = joint_delta * self.joints_vel_delta * self.sim_step

            # compute the joint angles we can actually go to
            new_joints = joint_delta + self.joints_sensor.joints_angles
            self.moveto_joints(new_joints, self.use_physics_sim)

        # returns execution time, gets used in gym env to log the times here
        return time() - cpu_epoch

    def set_trajectory_point(self, q):
        self.trajectory_point = q
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