from .engine_implementations import *
from .engine import Engine


class EngineRegistry:
    _engine_classes = {}

    @classmethod
    def get(cls, engine_type:str) -> Engine:
        try:
            return cls._engine_classes[engine_type]
        except KeyError:
            raise ValueError(f"unknown engine type : {engine_type}")

    @classmethod
    def register(cls, engine_type:str):
        def inner_wrapper(wrapped_class):
            cls._engine_classes[engine_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

EngineRegistry.register('Pybullet')(PybulletEngine)