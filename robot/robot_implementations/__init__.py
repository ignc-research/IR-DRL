# import the individual implementations
from .kr16 import KR16
from .ur5 import UR5
from robot.robot import Robot

# define the registry
class RobotRegistry:
    _robot_classes = {}

    @classmethod
    def get(cls, robot_type:str) -> Robot:
        try:
            return cls._robot_classes[robot_type]
        except KeyError:
            raise ValueError(f"unknown robot type : {robot_type}")

    @classmethod
    def register(cls, robot_type:str):
        def inner_wrapper(wrapped_class):
            cls._robot_classes[robot_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

# register the available classes
RobotRegistry.register('UR5')(UR5)
RobotRegistry.register('KR16')(KR16)