from .goal_implementations import *
from .goal import Goal


class GoalRegistry:
    _goal_classes = {}

    @classmethod
    def get(cls, goal_type:str, engine_type:str) -> Goal:
        try:
            return cls._goal_classes[(goal_type, engine_type)]
        except KeyError:
            raise ValueError(f"unknown goal type for {engine_type} : {goal_type}")

    @classmethod
    def register(cls, goal_type:str, engine_type:str):
        def inner_wrapper(wrapped_class):
            cls._goal_classes[(goal_type, engine_type)] = wrapped_class
            return wrapped_class
        return inner_wrapper

# Pybullet goals
GoalRegistry.register('PositionCollision', 'Pybullet')(PositionCollisionGoal_Pybullet)