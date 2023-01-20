from .lidar import LidarSensor
from .lidar_implementations import *
from .lidar_real import LidarSensorUR5Real

class LidarRegistry:
    _lidar_classes = {}

    @classmethod
    def get(cls, lidar_type:str) -> LidarSensor:
        try:
            return cls._lidar_classes[lidar_type]
        except KeyError:
            raise ValueError(f"unknown camera type : {lidar_type}")

    @classmethod
    def register(cls, lidar_type:str):
        def inner_wrapper(wrapped_class):
            cls._lidar_classes[lidar_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

LidarRegistry.register('LidarSensorUR5')(LidarSensorUR5)
LidarRegistry.register('LidarSensorUR5Real')(LidarSensorUR5Real)
