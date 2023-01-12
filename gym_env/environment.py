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
        self.show_auxillary_geometry = True
        self.train = False
        self.max_steps_per_episode = 1000

        # world attributes
        workspace_boundaries = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5]
        robot_base_positions = [np.array([0.0, -0.12, 0.5])]
        robot_base_orientations = [np.array([0, 0, 0, 1])]
        num_static_obstacles = 1
        num_moving_obstacles = 25
        box_measurements = [0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125]
        sphere_measurements = [0.005, 0.02]
        moving_obstacles_vels = [0.005, 0.025]
        #moving_obstacles_vels = [0.2, 0.2]
        moving_obstacles_directions = []
        moving_obstacles_trajectory_length = [0.3, 0.75]

        # robot attributes
        self.xyz_vels = [0.005]
        self.rpy_vels = [0.005]
        self.joint_vels = [0.005]
        self.joint_control = [False]

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

        # at this point we would generate all the sensors prescribed by the config for each robot and assign them to the robots
        # however, for now we simply generate the two necessary ones manually
        self.sensors = []
        ur5_1_position_sensor = PositionRotationSensor(self.normalize_sensor_data, True, ur5_1, 7)
        ur5_1_joint_sensor = JointsSensor(self.normalize_sensor_data, True, ur5_1)
        ur5_1.set_joint_sensor(ur5_1_joint_sensor)
        ur5_1.set_position_rotation_sensor(ur5_1_position_sensor)

        ur5_1_lidar_sensor = LidarSensorUR5(self.normalize_sensor_data, True, ur5_1, 20, 0, 0.3, 10, 6, True, True)
        self.sensors = [ur5_1_joint_sensor, ur5_1_position_sensor, ur5_1_lidar_sensor]


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

        self.world.register_robots(self.robots)

        # construct observation space from sensors and goals
        observation_space_dict = dict()
        for sensor in self.sensors:
            if sensor.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **sensor.get_observation_space_element()}  # merges the two dicts
        for goal in self.goals:
            if goal.add_to_observation_space:
                observation_space_dict = {**observation_space_dict, **goal.get_observation_space_element()}

        self.observation_space = gym.spaces.Dict(observation_space_dict)

        # construct action space from robots
        action_space_dims = 0
        for robot in self.robots:
            xyz_dims, joints_dims = robot.get_action_space_dims()
            if self.joint_control:
                action_space_dims += joints_dims
            else:
                xyz_dims += xyz_dims
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space_dims,), dtype=np.float32)

    def reset(self):

        for robot in self.robots:
            robot.build()

        ee_starting_points = self.world.create_ee_starting_points()
        position_targets = self.world.create_position_target()

        self.world.build()

        for sensor in self.sensors:
            sensor.reset()

        if self.show_auxillary_geometry:
            self.world.build_visual_aux()

    def step(self, action):
        self.world.update()
        for sensor in self.sensors:
            sensor.update()

    def _reward(self):
        pass

    