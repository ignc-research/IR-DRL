import pybullet as pyb
import gym
import numpy as np
from sensor.sensor import Sensor

class PositionRotationSensor(Sensor):

    def __init__(self, robot, normalize):

        super().__init__(robot, normalize)

    def update(self):
        pass

    def get_data(self):
        pass

    def get_observation_space_element(self):
        pass

    def get_secondary_data(self):
        pass