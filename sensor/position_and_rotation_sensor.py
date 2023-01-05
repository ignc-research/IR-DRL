import pybullet as pyb
import gym
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot
from time import time

class PositionRotationSensor(Sensor):

    def __init__(self, robot: Robot, normalize: bool, link_id: int, quaternion: bool=True):

        super().__init__(robot, normalize)

        # set output data field names
        self.output_name_rotation = "rotation_link_" + str(link_id) + "_" + self.robot.name
        # the position field is pointless as an absolute value and therefore not used in the get_data() method
        # the relative vector between target and current position will be added to the env observation space
        # by the position goal using this sensor's data (same goes for the rotation goal)
        self.output_name_position = "position_link_" + str(link_id) + "_" + self.robot.name
        self.output_name_velocity = "velocity_link_" + str(link_id) + "_" + self.robot.name

        # set the link of the robot for which data is to be gathered
        self.link_id = link_id

        # set whether the rotation is reported as quaternion or rpy
        self.quaternion = quaternion
        # set normalization constants (only needed if using rpy)
        if not self.quaternion:
            self.normalizing_constant_a = 2 / np.array([2*np.pi, 2*np.pi, 2*np.pi])  # pi is max, -pi is min
            self.normalizing_constant_b = np.ones(3) - np.multiply(self.normalizing_constant_a, np.array([np.pi, np.pi, np.pi]))

        # init data storage
        self.position = None
        self.position_prev = None
        self.rotation = None
        self.rotation
        self.position_velocity = None

    def update(self):
        new_time = time() - self.epoch
        self.pos_prev = self.position
        ee_link_state = pyb.getLinkState(self.robot.id, self.link_id, computeForwardKinematics=True)
        self.position = np.array(ee_link_state[4])
        self.rotation = ee_link_state[5]  # TODO: think about whether this maybe should be entry 1
        if not self.quaternion:
            self.rotation = pyb.getEulerFromQuaternion(self.rotation)
        self.rotation = np.array(self.rotation)
        self.position_velocity = (self.position - self.position_prev) / (new_time - self.time)
        self.time = time() - self.epoch

        return self.get_data()

    def get_data(self):
        if self.normalize:
            return self._normalize()
        else:
            return {self.output_name_rotation: self.rotation}

    def _normalize(self) -> dict:
        if self.quaternion:
            return {self.output_name_rotation: self.rotation}  # quaternions given by PyBullet are normalized by default
        else:
            return {self.output_name_rotation: np.multiply(self.normalizing_constant_a, self.rotation) + self.normalizing_constant_b}


    def get_observation_space_element(self):
        obs_sp_ele = dict()

        if self.quaternion:
            obs_sp_ele[self.output_name_rotation] = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        else:
            if self.normalize:
                obs_sp_ele[self.output_name_rotation] = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
            else:
                obs_sp_ele[self.output_name_rotation] = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)

    def get_secondary_data(self):
        logging_dict = dict()

        logging_dict[self.output_name_position] = self.position
        logging_dict[self.output_name_velocity] = self.position_velocity
        logging_dict[self.output_name_rotation] = self.rotation

        return logging_dict