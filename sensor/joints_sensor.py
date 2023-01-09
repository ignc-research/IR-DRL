import pybullet as pyb
import gym
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot
from time import time

class JointsSensor(Sensor):

    def __init__(self, robot: Robot, normalize: bool):

        super().__init__(normalize)
        
        # set associated robot
        self.robot = robot

        # set output data field name
        self.output_name = "joints_angles_" + self.robot.name + "_" + str(self.robot.id)

        # init data storage
        self.joints_dims = len(self.robot.joints_ids)
        self.joints_angles = None
        self.joints_angles_prev = None
        self.joints_velocities = None

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / self.robot.joints_range
        self.normalizing_constant_b = np.ones(6) - np.multiply(self.normalizing_constant_a, self.robot.joints_upper_limits)

        self.update()


    def update(self) -> dict:
        new_time = time() - self.epoch
        self.joints_angles_prev = self.joints_angles
        self.joints_angles = np.array([pyb.getJointState(self.robot.object_id, i)[0] for i in self.robot.joints_ids])
        self.joints_velocities = (self.joints_angles - self.joints_angles_prev) / (new_time - self.time)
        self.time = new_time

        return self.get_observation()

    def get_observation(self) -> dict:
        if self.normalize:
            return self._normalize()
        else:
            return {self.output_name: self.joints_angles}

    def _normalize(self) -> dict:
        return {self.output_name: np.multiply(self.normalizing_constant_a, self.joints_angles) + self.normalizing_constant_b}

    def get_observation_space_element(self) -> dict:
        
        if self.add_to_observation_space:
            obs_sp_ele = dict()

            if self.normalize:
                obs_sp_ele[self.output_name] = gym.spaces.Box(low=-1, high=1, shape=(self.joints_dims,), dtype=np.float32)
            else:
                obs_sp_ele[self.output_name] = gym.spaces.Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(self.joints_dims,), dtype=np.float32)

            return obs_sp_ele
        else:
            return {}

    def get_data_for_logging(self) -> dict:
        logging_dict = dict()

        logging_dict["joints_angles_" + self.robot_name] = self.joints_angles
        logging_dict["joints_velocities_" + self.robot_name] = self.joints_velocities

        return logging_dict

