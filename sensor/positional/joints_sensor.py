import pybullet as pyb
from gym.spaces import Box
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot
from time import time

__all__ = [
        'JointsSensor'
    ]

class JointsSensor(Sensor):

    def __init__(self, sensor_config):

        super().__init__(sensor_config)
        
        # set associated robot
        self.robot = sensor_config["robot"]

        # set output data field name
        self.output_name = "joints_angles_" + self.robot.name

        # init data storage
        self.joints_dims = len(self.robot.joints_limits_lower)
        self.joints_angles = None
        self.joints_angles_prev = None
        self.joints_velocities = None

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / self.robot.joints_range
        self.normalizing_constant_a = self.normalizing_constant_a.astype(np.float32)
        self.normalizing_constant_b = np.ones(6) - np.multiply(self.normalizing_constant_a, self.robot.joints_limits_upper)
        self.normalizing_constant_b = self.normalizing_constant_b.astype(np.float32)
        #self.update()


    def update(self, step) -> dict:
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            self.joints_angles_prev = self.joints_angles
            self.joints_angles = np.array([pyb.getJointState(self.robot.object_id, i)[0] for i in self.robot.joints_ids]).astype(np.float32)
            self.joints_velocities = (self.joints_angles - self.joints_angles_prev) / self.sim_step
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        self.joints_angles = np.array([pyb.getJointState(self.robot.object_id, i)[0] for i in self.robot.joints_ids]).astype(np.float32)
        self.joints_angles_prev = self.joints_angles
        self.joints_velocities = np.zeros(self.joints_dims)
        self.cpu_time = time() - self.cpu_epoch

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
                obs_sp_ele[self.output_name] = Box(low=-1, high=1, shape=(self.joints_dims,), dtype=np.float32)
            else:
                obs_sp_ele[self.output_name] = Box(low=np.float32(self.robot.joints_limits_lower), high=np.float32(self.robot.joints_limits_upper), shape=(self.joints_dims,), dtype=np.float32)

            return obs_sp_ele
        else:
            return {}

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict["joints_angles_" + self.robot.name] = self.joints_angles
        logging_dict["joints_velocities_" + self.robot.name] = self.joints_velocities
        logging_dict["joints_sensor_cpu_time_" + self.robot.name] = self.cpu_time

        return logging_dict

