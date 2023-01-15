import gym
import numpy as np
import pybullet as pyb

# import abstracts
from robot.robot import Robot
from sensor.sensor import Sensor
from goal.goal import Goal
from world.world import World

# import implementations, new ones hav to be added here
#   worlds
from world.random_obstacles import RandomObstacleWorld
#   robots
from robot.ur5 import UR5
#   sensors
from sensor.joints_sensor import JointsSensor
from sensor.position_and_rotation_sensor import PositionRotationSensor
from sensor.lidar import LidarSensorUR5
#   goals
from goal.position_collision import PositionCollisionGoal

class ModularDRLEnv(gym.Env):

    def __init__(self, env_config):

        # here the parser for the env_config (a python dict for the external YAML file) will appear at some point
        # for now, the attributes needed to get this prototype to run are set manually here
        
        # general env attributes
        self.normalize_sensor_data = False
        self.normalize_rewards = False
        self.display = True
        self.show_auxillary_geometry_world = True
        self.show_auxillary_geometry_goal = False
        self.train = False
        self.max_steps_per_episode = 1000

        # tracking variables
        self.steps_current_episode = 0

        # world attributes
        workspace_boundaries = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5]
        robot_base_positions = [np.array([0.0, -0.12, 0.5]), np.array([0.0, 1.12, 0.5])]
        robot_base_orientations = [np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1])]
        num_static_obstacles = 4
        num_moving_obstacles = 2
        box_measurements = [0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125]
        sphere_measurements = [0.005, 0.02]
        moving_obstacles_vels = [0.005, 0.025]
        #moving_obstacles_vels = [0.2, 0.2]
        moving_obstacles_directions = []
        moving_obstacles_trajectory_length = [0.3, 0.75]

        # robot attributes
        self.xyz_vels = [0.005, 0.005]
        self.rpy_vels = [0.005, 0.005]
        self.joint_vels = [0.015, 0.015]
        self.joint_control = [False, False]

        # set up the PyBullet client
        disp = pyb.DIRECT if not self.display else pyb.GUI
        pyb.connect(disp)
        pyb.setAdditionalSearchPath("./assets/")
        
        self.world = RandomObstacleWorld(workspace_boundaries=workspace_boundaries,
                                         robot_base_positions=robot_base_positions,
                                         robot_base_orientations=robot_base_orientations,
                                         num_static_obstacles=num_static_obstacles,
                                         num_moving_obstacles=num_moving_obstacles,
                                         box_measurements=box_measurements,
                                         sphere_measurements=sphere_measurements,
                                         moving_obstacles_vels=moving_obstacles_vels,
                                         moving_obstacles_directions=moving_obstacles_directions,
                                         moving_obstacles_trajectory_length=moving_obstacles_trajectory_length)

        # at this point robots would dynamically be created as needed by the config/the world
        # however, for now we generate one manually
        self.robots = []
        ur5_1 = UR5(name="HAL9000", 
                   world=self.world,
                   base_position=robot_base_positions[0],
                   base_orientation=robot_base_orientations[0],
                   resting_angles=np.array([np.pi/2, -np.pi/6, -2*np.pi/3, -4*np.pi/9, np.pi/2, 0.0]),
                   end_effector_link_id=7,
                   base_link_id=1,
                   control_joints=self.joint_control[0],
                   xyz_vel=self.xyz_vels[0],
                   rpy_vel=self.rpy_vels[0],
                   joint_vel=self.joint_vels[0])
        self.robots.append(ur5_1)
        ur5_1.id = 1

        ur5_2 = UR5(name="HAL9050", 
                   world=self.world,
                   base_position=robot_base_positions[1],
                   base_orientation=robot_base_orientations[1],
                   resting_angles=np.array([-np.pi/2, -np.pi/6, -2*np.pi/3, -4*np.pi/9, np.pi/2, 0.0]),
                   end_effector_link_id=7,
                   base_link_id=1,
                   control_joints=self.joint_control[1],
                   xyz_vel=self.xyz_vels[1],
                   rpy_vel=self.rpy_vels[1],
                   joint_vel=self.joint_vels[1])
        self.robots.append(ur5_2)
        ur5_2.id = 2

        # at this point we would generate all the sensors prescribed by the config for each robot and assign them to the robots
        # however, for now we simply generate the two necessary ones manually
        self.sensors = []
        ur5_1_position_sensor = PositionRotationSensor(self.normalize_sensor_data, True, ur5_1, 7)
        ur5_1_joint_sensor = JointsSensor(self.normalize_sensor_data, True, ur5_1)
        ur5_1.set_joint_sensor(ur5_1_joint_sensor)
        ur5_1.set_position_rotation_sensor(ur5_1_position_sensor)

        ur5_1_lidar_sensor = LidarSensorUR5(self.normalize_sensor_data, True, ur5_1, 20, 0, 0.3, 10, 6, True, True)

        ur5_2_position_sensor = PositionRotationSensor(self.normalize_sensor_data, True, ur5_2, 7)
        ur5_2_joint_sensor = JointsSensor(self.normalize_sensor_data, True, ur5_2)
        ur5_2.set_joint_sensor(ur5_2_joint_sensor)
        ur5_2.set_position_rotation_sensor(ur5_2_position_sensor)

        ur5_2_lidar_sensor = LidarSensorUR5(self.normalize_sensor_data, True, ur5_2, 20, 0, 0.3, 10, 6, True, True)

        self.sensors = [ur5_1_joint_sensor, ur5_1_position_sensor, ur5_1_lidar_sensor, ur5_2_position_sensor, ur5_2_joint_sensor, ur5_2_lidar_sensor]


        # at this point we would generate all the goals needed and assign them to their respective robots
        # however, for the moment we simply generate the one we want for testing
        self.goals = []
        ur5_1_goal = PositionCollisionGoal(robot=ur5_1,
                                           normalize_rewards=self.normalize_rewards,
                                           normalize_observations=self.normalize_sensor_data,
                                           train=self.train,
                                           max_steps=self.max_steps_per_episode,
                                           reward_success=10,
                                           reward_collision=-10,
                                           reward_distance_mult=-0.01,
                                           dist_threshold_start=0.3,
                                           dist_threshold_end=0.01,
                                           dist_threshold_increment_start=0.01,
                                           dist_threshold_increment_end=0.001)
        self.goals.append(ur5_1_goal)
        ur5_1.set_goal(ur5_1_goal)

        ur5_2_goal = PositionCollisionGoal(robot=ur5_2,
                                           normalize_rewards=self.normalize_rewards,
                                           normalize_observations=self.normalize_sensor_data,
                                           train=self.train,
                                           max_steps=self.max_steps_per_episode,
                                           reward_success=10,
                                           reward_collision=-10,
                                           reward_distance_mult=-0.01,
                                           dist_threshold_start=0.3,
                                           dist_threshold_end=0.01,
                                           dist_threshold_increment_start=0.01,
                                           dist_threshold_increment_end=0.001)
        self.goals.append(ur5_2_goal)
        ur5_2.set_goal(ur5_2_goal)

        self.world.register_robots(self.robots)

        # construct observation space from sensors and goals
        # each sensor and goal will add elements to the observation space with fitting names
        observation_space_dict = dict()
        for sensor in self.sensors:
            if sensor.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **sensor.get_observation_space_element()}  # merges the two dicts
        for goal in self.goals:
            if goal.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **goal.get_observation_space_element()}

        self.observation_space = gym.spaces.Dict(observation_space_dict)

        # construct action space from robots
        # the action space will be a vector with the length of all robot's control dimensions added up
        # e.g. if one robot needs 4 values for its control and another 10,
        # the action space will be a 10-vector with the first 4 elements working for robot 1 and the last 6 for robot 2
        self.action_space_dims = []
        for idx, robot in enumerate(self.robots):
            ik_dims, joints_dims = robot.get_action_space_dims()
            if self.joint_control[idx]:
                self.action_space_dims.append(joints_dims)
            else:
                self.action_space_dims.append(ik_dims)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(sum(self.action_space_dims),), dtype=np.float32)

    def reset(self):

        # reset the tracking variables
        self.steps_current_episode = 0

        # spawn robots in world
        for robot in self.robots:
            robot.build()

    	# get a set of starting positions for the end effectors
        ee_starting_points = self.world.create_ee_starting_points()
        
        # get position and roation goals
        position_targets = self.world.create_position_target()
        rotation_targets = self.world.create_rotation_target()

        # spawn world objects
        self.world.build()

        # set the robots into the starting positions
        for idx, ee_pos in enumerate(ee_starting_points):
            if ee_pos[0] is None:
                continue  # nothing to do here
            elif ee_pos[1] is None:
                # only position
                self.robots[idx].moveto_xyz(ee_pos[0])
            else:
                # both position and rotation
                self.robots[idx].moveto_xyzquat(ee_pos[0], ee_pos[1])

        for sensor in self.sensors:
            sensor.reset()

        # render non-essential visual stuff
        if self.show_auxillary_geometry_world:
            self.world.build_visual_aux()
        if True:
            for goal in self.goals:
                goal.build_visual_aux()

        return self._get_obs()

    def _get_obs(self):
        obs_dict = dict()
        # get the sensor data
        for sensor in self.sensors:
            if sensor.add_to_observation_space:
                obs_dict = {**obs_dict, **sensor.get_observation()}
        for goal in self.goals:
            if goal.add_to_observation_space:
                obs_dict = {**obs_dict, **goal.get_observation()}

        # no normalizing here, that should be handled by the sensors and goals

        return obs_dict

    def step(self, action):
        
        # convert to numpy
        action = np.array(action)
        
        # update world
        self.world.update()

        # apply the action to all robots that have to be moved
        offset = 0  # the offset at which the ith robot sits in the action array
        for idx, robot in enumerate(self.robots):
            current_robot_action = action[offset : self.action_space_dims[idx] + offset]
            offset += self.action_space_dims[idx]
            robot.process_action(current_robot_action)

        # update the sensor data
        for sensor in self.sensors:
            sensor.update()

        # update the collision model
        self.world.perform_collision_check()

        # calculate reward and get termination conditions
        rewards = []
        dones = []
        successes = []
        for goal in self.goals:
            reward_info = goal.reward(self.steps_current_episode)  # tuple: reward, success, done
            rewards.append(reward_info[0])
            successes.append(reward_info[1])
            dones.append(reward_info[2])
        # determine overall env termination condition
        done = np.average(dones) > 0  # one done out of all goals/robots suffices for the entire env to be done
        success = np.average(successes) == 1  # all goals must be succesful for the entire env to be
        # reward
        # if we are normalizing the reward, we must also account for the number of robots 
        # (each goal will output a reward from -1 to 1, so e.g. three robots would have a cumulative reward range from -3 to 3)
        if self.normalize_rewards:
            reward = np.average(reward)
        # otherwise we can just add the single rewards up
        else:
            reward = np.sum(rewards)

        # update tracking variables
        self.steps_current_episode += 1

        info = {}

        return self._get_obs(), reward, done, info

    def _reward(self):
        pass

    