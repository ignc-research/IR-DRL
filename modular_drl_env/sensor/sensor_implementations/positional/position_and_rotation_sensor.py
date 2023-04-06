from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from modular_drl_env.util.quaternion_util import quaternion_to_rpy
from time import time
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
        'PositionRotationSensor'
    ]

class PositionRotationSensor(Sensor):

    def __init__(self, normalize: bool, add_to_observation_space:bool, add_to_logging: bool, sim_step: float, update_steps: int, sim_steps_per_env_step: int, robot: Robot, link_id: int, quaternion: bool=True):

        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, sim_steps_per_env_step)

        # WARNING: this position sensor will not return the position as part of the observation space
        # because absolute position is not useful for the model
        # its data will be used by a Position goal to construct a relative vector
        # the robot position is still stored as a class attribute here

        # set associated robot
        self.robot = robot

        # set output data field names
        self.output_name_rotation = "rotation_" + str(link_id) + "_" + self.robot.name
        # the position field is pointless as an absolute value and therefore not used in the get_data() method
        # the relative vector between target and current position will be added to the env observation space
        # by the position goal using this sensor's data (same goes for the rotation goal)
        self.output_name_position = "position_" + str(link_id) + "_" + self.robot.name
        self.output_name_velocity = "velocity_" + str(link_id) + "_" + self.robot.name
        self.output_name_time = "position_rotation_sensor_cpu_time_" + self.robot.name

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
        self.position_velocity = np.zeros(3)

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            self.position_prev = self.position
            self.position, self.rotation, _, _ = pyb_u.get_link_state(self.robot.object_id, self.link_id)
            if not self.quaternion:
                self.rotation = quaternion_to_rpy(self.rotation)
            self.position_velocity = (self.position - self.position_prev) / (self.update_steps * self.sim_step * self.sim_steps_per_env_step)
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        self.position, self.rotation, _, _ = pyb_u.get_link_state(self.robot.object_id, self.link_id)
        self.position_prev = self.position
        if not self.quaternion:
            self.rotation = quaternion_to_rpy(self.rotation)
        self.position_velocity = np.zeros(3)
        self.cpu_time = time() - self.cpu_epoch

    def get_observation(self):
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

        if self.add_to_observation_space:
            obs_sp_ele = dict()

            if self.quaternion:
                obs_sp_ele[self.output_name_rotation] = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            else:
                if self.normalize:
                    obs_sp_ele[self.output_name_rotation] = Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
                else:
                    obs_sp_ele[self.output_name_rotation] = Box(low=np.float32(-np.pi), high=np.float32(np.pi), shape=(3,), dtype=np.float32)
            return obs_sp_ele
        else:
            return {}

    def get_data_for_logging(self):
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict[self.output_name_position] = self.position
        logging_dict[self.output_name_velocity] = self.position_velocity
        logging_dict[self.output_name_rotation] = self.rotation
        logging_dict[self.output_name_time] = self.cpu_time

        return logging_dict