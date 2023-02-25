from .sensor import Sensor
from .sensor_implementations import *

class SensorRegistry:
    _sensor_classes = {}

    @classmethod
    def get(cls, sensor_type:str, engine_type: str) -> Sensor:
        try:
            return cls._sensor_classes[(sensor_type, engine_type)]
        except KeyError:
            raise ValueError(f"unknown sensor type : {sensor_type}")

    @classmethod
    def register(cls, sensor_type:str, engine_type: str):
        def inner_wrapper(wrapped_class):
            cls._sensor_classes[(sensor_type, engine_type)] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet sensors
SensorRegistry.register('PositionRotation', 'Pybullet')(PositionRotationSensor_Pybullet)
SensorRegistry.register('Joints', 'Pybullet')(JointsSensor_Pybullet)
SensorRegistry.register('Obstacle', 'Pybullet')(ObstacleSensor_Pybullet)
SensorRegistry.register('LidarSensorUR5', 'Pybullet')(LidarSensorUR5_Pybullet)
SensorRegistry.register('LidarSensorUR5Explainable', 'Pybullet')(LidarSensorUR5_Explainable_Pybullet)
SensorRegistry.register('LidarSensorKR16', 'Pybullet')(LidarSensorKR16_Pybullet)
SensorRegistry.register('OnBodyUR5', 'Pybullet')(OnBodyCameraUR5_Pybullet)
SensorRegistry.register('Floating', 'Pybullet')(StaticFloatingCamera_Pybullet)
SensorRegistry.register('FloatingFollowEffector', 'Pybullet')(StaticFloatingCameraFollowEffector_Pybullet)
SensorRegistry.register('BuddyRobotCamera', 'Pybullet')(BuddyRobotCamera_Pybullet)