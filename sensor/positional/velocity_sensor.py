import pybullet as pyb
from gym.spaces import Box
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot
from time import time


class VelocitySensor(Sensor):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)

        # set robot
        self.robot = sensor_config["robot"]

        # velocities
        self.vels : np.array

    def update(self, step) -> dict:
        self.cpu_epoch = time()
        vels = []
        for i in self.robot.joints_ids:
            vels.append(pyb.getJointState(self.robot.object_id, i)[1])
        self.vels = np.asarray(vels, dtype=np.float32)
        self.cpu_time = time() - self.cpu_epoch
        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        vels = []
        for i in self.robot.joints_ids:
            vels.append(pyb.getJointState(self.robot.object_id, i)[1])
        self.vels = np.asarray(vels, dtype=np.float32)
        self.cpu_time = time() - self.cpu_epoch

    def get_observation(self):
        if self.normalize:
            return self._normalize()
        else:
            return {"angular_velocities": self.vels}

    def _normalize(self) -> dict:
        return {"angular_velocities": self.vels / 10}

    def get_observation_space_element(self):
        if self.add_to_observation_space:
            obs_sp_ele = dict()
            if self.normalize:
                obs_sp_ele["angular_velocities"] = Box(low=-1, high=1, shape=(6,), dtype=np.float32)
            else:
                obs_sp_ele["angular_velocities"] = Box(low=-10, high=10, shape=(6,), dtype=np.float32)
            return obs_sp_ele
        else:
            return {}

    def get_data_for_logging(self):
        if not self.add_to_logging:
            return {}
        logging_dict = dict()
        logging_dict["angular_velocities"] = self.vels
        return logging_dict

