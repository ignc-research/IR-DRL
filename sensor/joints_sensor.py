import pybullet as pyb
import gym
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot

class JointSensor(Sensor):

    def __init__(self, robot: Robot, normalize: bool):

        super().__init__(robot, normalize)
        
        # init data storage
        self.joints_dims = len(self.robot.joints_ids)
        self.joints_angles = None
        self.joints_velocities = None

        self.update()


    def update(self) -> dict:
        pass

    def get_data(self) -> dict:
        pass

    def _normalize(self) -> dict:
        pass

    def get_observation_space_element(self) -> dict:
        
        obs_sp_ele = dict()
        ele_name = "joints_angles_" + self.robot.name

        if self.normalize:
            obs_sp_ele[ele_name] = gym.spaces.Box(low=-1, high=1, shape=(self.joints_dims,), dtype=np.float32)
        else:
            obs_sp_ele[ele_name] = gym.spaces.Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(self.joints_dims,), dtype=np.float32)

        return obs_sp_ele

    def get_data_for_logging(self) -> dict:
        
        logging_dict = dict()
        logging_dict["joints_angles_" + self.robot_name] = self.joints_angles
        logging_dict["joints_velocities_" + self.robot_name] = self.joints_velocities

        