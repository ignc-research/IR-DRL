from .robot_implementations import *
from .robot import Robot


class RobotRegistry:
    _robot_classes = {}

    @classmethod
    def get(cls, robot_type:str, engine_type:str) -> Robot:
        try:
            return cls._robot_classes[(robot_type, engine_type)]
        except KeyError:
            raise ValueError(f"unknown robot type for engine {engine_type} : {robot_type}")

    @classmethod
    def register(cls, robot_type:str, engine_type:str):
        def inner_wrapper(wrapped_class):
            cls._robot_classes[(robot_type, engine_type)] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet robots
RobotRegistry.register('UR5', 'Pybullet')(UR5_Pybullet)
RobotRegistry.register('UR5_RRT', 'Pybullet')(UR5_RRT_Pybullet)
RobotRegistry.register('KR16', 'Pybullet')(KR16_Pybullet)
RobotRegistry.register('Kukaiiwa', 'Pybullet')(Kukaiiwa_Pybullet)