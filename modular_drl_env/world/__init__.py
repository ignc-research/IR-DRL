from .world_implementations import *
from .world import World


class WorldRegistry:
    _world_classes = {}

    @classmethod
    def get(cls, world_type:str, engine_type:str) -> World:
        try:
            return cls._world_classes[(world_type, engine_type)]
        except KeyError:
            raise ValueError(f"unknown world type for {engine_type}: {world_type}")

    @classmethod
    def register(cls, world_type:str, engine_type:str):
        def inner_wrapper(wrapped_class):
            cls._world_classes[(world_type, engine_type)] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet worlds
WorldRegistry.register('RandomObstacle', 'Pybullet')(RandomObstacleWorld)
WorldRegistry.register('Testcases', 'Pybullet')(TestcasesWorld)
WorldRegistry.register('TableExperiment', 'Pybullet')(TableExperiment)
WorldRegistry.register('Generated', 'Pybullet')(GeneratedWorld)
WorldRegistry.register('KukaShelfExperiment', 'Pybullet')(KukaShelfExperiment)