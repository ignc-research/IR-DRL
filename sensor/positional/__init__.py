from .joints_sensor import JointsSensor
from .position_and_rotation_sensor import PositionRotationSensor
from typing import Union


class PositionalRegistry:
    _positional_classes = {}

    @classmethod
    def get(cls, positional_type:str) -> Union[JointsSensor, PositionRotationSensor]:
        try:
            return cls._positional_classes[positional_type]
        except KeyError:
            raise ValueError(f"unknown positional type : {positional_type}")

    @classmethod
    def register(cls, positional_type:str):
        def inner_wrapper(wrapped_class):
            cls._positional_classes[positional_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

PositionalRegistry.register('PositionRotation')(PositionRotationSensor)
PositionalRegistry.register('Joints')(JointsSensor)