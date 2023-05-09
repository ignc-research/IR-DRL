from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from modular_drl_env.world.world import World
from modular_drl_env.util.quaternion_util import quaternion_to_rpy, rpy_to_quaternion
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet as pyb
from time import process_time

CONTROL_MODES = [
    "inverse_kinematics",
    "joint_positions",
    "joint_velocities",
    "joint_target"
]

class Robot(ABC):
    """
    Abstract Base Class for a robot. Methods signed with abstractmethod need to be implemented by subclasses.
    Movement is already implemented and should work if all the class variables are set correctly.
    See the ur5 robot for examples.
    """

    def __init__(self, name: str,
                       id_num: int,
                       world: World,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       jt_joint_delta: float=0.5,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[],
                       self_collision: bool=True):
        super().__init__()

        # set name
        self.name = name

        # URDF file path, NOTE: this and two variables below are literally the only things that you have to set for your implementation
        self.urdf_path = None
        # link ids, these have to be set in your subclass!
        self.end_effector_link_id = None
        self.base_link_id = None
        
        # set id field, this will be given by the world containing this robot
        # it's used by other objects such as goals to access the correct robot's data when it's in some list somewhere
        self.mgt_id = id_num

        # set world
        self.world = world

        # set sim step
        self.sim_step = sim_step

        # base position
        self.base_position = np.array(base_position)

        # base orientation
        self.base_orientation = np.array(base_orientation)

        # resting pose angles
        self.resting_pose_angles = np.array(resting_angles)

        # use physics sim or simply teleport for movement
        self.use_physics_sim = use_physics_sim       

        # PyBullet and URDF related variables
        self.urdf_path = None  # set in subclass, should be the relative path to the robot's URDF file
        self.object_id = None  # str handle for unique identification with our pybullet handler, will be something like robot_0
        self.controlled_joints_ids = []  # array of controleld actionable joint ids, this gets filled by build
        self.all_joints_ids = []  # all actionable joint ids
        self.joints_limits_lower = []  # this and the two below are set by the build method
        self.joints_limits_upper = []  
        self.joints_range = None
        self.joints_max_velocities = None  # again, set in build method
        self.joints_max_forces = None  # same as the one above

        # control mode
        if type(control_mode) == str:
            assert control_mode in CONTROL_MODES, "[ROBOT init] unknown control mode!"
            self.control_mode = CONTROL_MODES.index(control_mode)
        else:
            assert control_mode >= 0 and control_mode < 4, "[ROBOT init] unknown control mode!"
            self.control_mode = control_mode

        # goal associated with the robot
        self.goal = None

        # sensors associated with the robot
        self.sensors = []
        # joint and position sensor (for end effector) are mandatory and thus treated separately
        self.joints_sensor = None
        self.position_rotation_sensor = None

        # in inverse kinmematics mode limits the maximum desired movement per step to this
        # note: independent of this, movements will still be limited by maximum joint velocities and joint position limits
        self.xyz_delta = ik_xyz_delta
        self.rpy_delta = ik_rpy_delta

        # for control mode 3
        # the amount of deviation from the set joint target that is allowed
        self.joint_delta = jt_joint_delta

        # if float: a multiplier for all joint velocities
        # if list: overwrite for max joint velocities, has to overwrite every single one
        self.joint_velocities_overwrite = joint_velocities_overwrite
        # same as for the velocities, just for joint positions
        self.joint_limits_overwrite = joint_limits_overwrite

        # these are the controlled joints, if this list is empty (default case) then all controllabe joints will be controlled by the agent
        self.controlled_joints = controlled_joints

        # bool for whether robot will have self collisions
        self.self_collision = self_collision

        # joint angles for control mode 3, only ever gets set from the outside by other entities, e.g. a goal
        self.control_target = []

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
        return (len(self.controlled_joints_ids), 6)

    def build(self) -> None:
        """
        Method that spawns the robot into the simulation, moves its base to the desired position and orientation
        and sets its joints to the resting angles. Also populates the PyBullet variables with information.
        """
        self.object_id = pyb_u.load_urdf(urdf_path=self.urdf_path, position=self.base_position, orientation=self.base_orientation, is_robot=True, self_collisions=self.self_collision)
        self.all_joints_ids = pyb_u.get_controllable_joint_ids(self.object_id)
        self.all_joints_ids = [ele[0] for ele in self.all_joints_ids]
        if self.controlled_joints:
            self.controlled_joints_ids = self.controlled_joints
        else:
            self.controlled_joints_ids = self.all_joints_ids
        self.indices_controlled = []
        for idx, joint_id in enumerate(self.all_joints_ids):
            if joint_id in self.controlled_joints_ids:
                self.indices_controlled.append(idx)
        self.indices_controlled = np.array(self.indices_controlled)

        # handle the limit overwrite input
        self.joint_limits_overwrite = self.joint_limits_overwrite if type(self.joint_limits_overwrite) == list else [self.joint_limits_overwrite for _ in self.all_joints_ids]
        self.joint_velocities_overwrite = self.joint_velocities_overwrite if type(self.joint_velocities_overwrite) == list else [self.joint_velocities_overwrite for _ in self.all_joints_ids]

        # get info about joint limits, forces and max velocities
        lowers = []
        uppers = []
        forces = []
        velos = []
        for idx, joint_id in enumerate(self.all_joints_ids):
            lower, upper, force, velocity = pyb_u.get_joint_dynamics(self.object_id, joint_id)
            lowers.append(lower * self.joint_limits_overwrite[idx])
            uppers.append(upper * self.joint_limits_overwrite[idx])
            forces.append(force)
            velos.append(velocity * self.joint_velocities_overwrite[idx])
        # set the internal full joint attributes
        self._joints_limits_lower = np.array(lowers)
        self._joints_limits_upper = np.array(uppers)
        self._joints_range = self._joints_limits_upper - self._joints_limits_lower
        self._joints_max_forces = np.array(forces)
        self._joints_max_velocities = np.array(velos)
        self._resting_pose_angles = self.resting_pose_angles
        # set the exposed subset of controlled joint attributes
        self.joints_limits_lower = self._joints_limits_lower[self.indices_controlled]
        self.joints_limits_upper = self._joints_limits_upper[self.indices_controlled]
        self.joints_range = self._joints_range[self.indices_controlled]
        self.joints_max_forces = self._joints_max_forces[self.indices_controlled]
        self.joints_max_velocities = self._joints_max_velocities[self.indices_controlled]
        self.resting_pose_angles = self.resting_pose_angles[self.indices_controlled]

        # modify the internal Pybullet representation to obey our new limits on position and velocity
        for idx, joint_id in enumerate(self.all_joints_ids):
            pyb_u.set_joint_dynamics(self.object_id, joint_id, self._joints_max_velocities[idx], self._joints_limits_lower[idx], self._joints_limits_upper[idx])

        self.moveto_joints(self._resting_pose_angles, False, self.all_joints_ids)

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
        cpu_epoch = process_time()
        if self.control_mode == 0:  
            # control via inverse kinematics:
            # actions are small changes in xyz and rpy of the robot's end effector
            # we calculate the changed position, then use inverse kinematics to get the equivalent joint angles
            # then we apply those
            pos_delta = action[:3] * self.xyz_delta
            rpy_delta = action[3:] * self.rpy_delta

            new_pos = self.position_rotation_sensor.position + pos_delta
            new_rpy = quaternion_to_rpy(self.position_rotation_sensor.rotation) + rpy_delta

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
                joint_dist = np.linalg.norm(joint_delta)
                joint_dist = joint_dist if joint_dist != 0 else 1
                joint_delta = joint_delta / joint_dist
                step_times_velocity = np.min(self.joints_max_velocities) * self.sim_step
                if joint_dist > step_times_velocity:
                    joint_mul = step_times_velocity
                else:
                    joint_mul = joint_dist
                joint_delta = joint_delta * joint_mul
                # compute the joint angles we can actually go to
                new_joints = joint_delta + self.joints_sensor.joints_angles

            # execute movement
            self.moveto_joints(new_joints, self.use_physics_sim)

        elif self.control_mode == 2:  
            # control via joint velocities
            # actions are joint velocities
            # if we use the physics sim, the engine can deal with those on its own
            # if we don't, we run simple algebra to get the new joint angles for this step and then apply them

            # transform action (-1 to 1) to joint velocities
            new_joint_vels = action * self.joints_max_velocities[self.indices_controlled]
            if not self.use_physics_sim:
                # compute the delta for this sim step
                joint_delta = new_joint_vels * self.sim_step
                # add the delta to current joint angles
                new_joints = joint_delta + self.joints_sensor.joints_angles[self.indices_controlled]
                # execute movement
                self.moveto_joints(new_joints, False)

            else:
                # use engine to apply these velocities to robot
                self.moveto_joints_vels(new_joint_vels)
        
        # returns execution time, gets used in gym env to log the times here
        elif self.control_mode == 3:
            # control via a pre-set joint angle target
            # in this control mode we have joint angles and actions correspond to deviations from it
            # the robot will then try to move to set joint angles + deviation
            # e.g. a zero action will just be the trajectory angle

            # convert action to desired joint angles
            new_joints = self.control_target + self.joint_delta * action
            # execute movement
            self.moveto_joints(new_joints, self.use_physics_sim)


        return process_time() - cpu_epoch

    def moveto_joints_vels(self, desired_joints_velocities: np.ndarray):
        """
        Uses the actual physics simulation to set the torques in the robot's actuator such that they result in the desired joint velocities.

        :param desired_joints_velocities: Vector containing the new joint velocities.
        """
        pyb_u.set_joint_targets(
            robot_id=self.object_id,
            joint_ids=self.controlled_joints_ids,
            velocity=desired_joints_velocities,
            forces=self.joints_max_forces)

    def moveto_joints(self, desired_joints_angles: np.ndarray, use_physics_sim: bool, joints_ids: list=None):
        """
        Moves the robot's joints towards the desired configuration.
        Also automatically clips the input such that no joint limits are violated.

        :param desired_joints_angles: Vector containing the desired new joint angles
        """
        if joints_ids is None:
            joints_ids = self.controlled_joints_ids

            # clip desired angles at max/min
            upper_limit_mask = desired_joints_angles > self.joints_limits_upper
            lower_limit_mask = desired_joints_angles < self.joints_limits_lower
            desired_joints_angles[upper_limit_mask] = self.joints_limits_upper[upper_limit_mask]
            desired_joints_angles[lower_limit_mask] = self.joints_limits_lower[lower_limit_mask]
        else:
            indices = [self.all_joints_ids.index(i) for i in joints_ids]
            upper_limit_mask = desired_joints_angles > self._joints_limits_upper[indices]
            lower_limit_mask = desired_joints_angles < self._joints_limits_lower[indices]
            desired_joints_angles[upper_limit_mask] = self._joints_limits_upper[indices][upper_limit_mask]
            desired_joints_angles[lower_limit_mask] = self._joints_limits_lower[indices][lower_limit_mask]

        # apply movement
        if use_physics_sim:
            pyb_u.set_joint_targets(
                robot_id=self.object_id,
                joint_ids=joints_ids,
                position=desired_joints_angles,
                forces=self.joints_max_forces
            )
        else:
            pyb_u.set_joint_states(
                robot_id=self.object_id,
                joint_ids=joints_ids,
                position=desired_joints_angles
            )

    def moveto_xyzrpy(self, desired_xyz: np.ndarray, desired_rpy: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position and rpy orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_rpy: Vector containing the desired new rpy orientation of the end effector.
        """
        desired_quat = rpy_to_quaternion(desired_rpy)
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints, use_physics_sim)

    def moveto_xyzquat(self, desired_xyz: np.ndarray, desired_quat: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position and quat orientation.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        :param desired_quat: Vector containing the desired new quaternion orientation of the end effector.
        """
        joints = self._solve_ik(desired_xyz, desired_quat)
        self.moveto_joints(joints, use_physics_sim, self.all_joints_ids)

    def moveto_xyz(self, desired_xyz: np.ndarray, use_physics_sim: bool):
        """
        Moves the robot such that end effector is in the desired xyz position.
        Orientation will not be controlled.

        :param desired_xyz: Vector containing the desired new xyz position of the end effector.
        """
        joints = self._solve_ik(desired_xyz, None)
        self.moveto_joints(joints, use_physics_sim, self.all_joints_ids)

    def _solve_ik(self, xyz: np.ndarray, quat:Union[np.ndarray, None]):
        """
        Solves the robot's inverse kinematics for the desired pose.
        Returns the joint angles required

        :param xyz: Vector containing the desired xyz position of the end effector.
        :param quat: Vector containing the desired rotation of the end effector.
        :return: Vector containing the joint angles required to reach the pose.
        """
        return pyb_u.solve_inverse_kinematics(
            robot_id=self.object_id,
            link_id=self.end_effector_link_id,
            target_position=xyz,
            target_orientation=quat
        )

    def move_base(self, desired_base_position: np.ndarray, desired_base_orientation: np.ndarray):
        """
        Moves the base of the robot towards the desired position and orientation.

        :param desired_base_position: Vector containing the desired xyz position of the base.
        :param desired_base_orientation: Vector containing the desired rotation of the base.
        """

        self.base_position = desired_base_position
        self.base_orientation = desired_base_orientation
        pyb_u.set_base_pos_and_ori(self.object_id, desired_base_position, desired_base_orientation)

    def sample_valid_configuration(self, only_controlled_joints=True):
        """
        Samples the configuration space for a random element that is not in self-collision, does not check for collision with objects!
        """
        if only_controlled_joints:
            l = self.joints_limits_lower
            u = self.joints_limits_upper
            d = len(self.controlled_joints_ids)
            ids = self.controlled_joints_ids
        else:
            l = self._joints_limits_lower
            u = self._joints_limits_upper
            d = len(self.all_joints_ids)
            ids = self.all_joints_ids
        while True:
            sample = np.random.uniform(low=l, high=u, size=(d,))
            self.moveto_joints(sample, False, ids)
            pyb.performCollisionDetection()
            contacts = pyb.getContactPoints(pyb_u.to_pb(self.object_id))
            self_col = False
            for contact in contacts:
                if contact[1] == pyb_u.to_pb(self.object_id) and contact[2] == pyb_u.to_pb(self.object_id):
                    self_col = True
                    break
            if not self_col:
                break # if we reach this line, there were no self collisions
        self.moveto_joints(self.resting_pose_angles, False, self.controlled_joints_ids)
        return sample
