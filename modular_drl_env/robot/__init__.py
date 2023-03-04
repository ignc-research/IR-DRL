from .robot_implementations import *
from .robot import Robot


class RobotRegistry:
    _robot_classes = {}

    @classmethod
    def get(cls, robot_type:str) -> Robot:
        try:
            return cls._robot_classes[robot_type]
        except KeyError:
            raise ValueError(f"unknown robot type: {robot_type}")

    @classmethod
    def register(cls, robot_type:str):
        def inner_wrapper(wrapped_class):
            cls._robot_classes[robot_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet robots
RobotRegistry.register('UR5')(UR5)
RobotRegistry.register('UR5_RRT')(UR5_RRT)
RobotRegistry.register('KR16')(KR16)
RobotRegistry.register('Kukaiiwa')(Kukaiiwa)