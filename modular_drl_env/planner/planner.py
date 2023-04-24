from abc import ABC, abstractmethod
from modular_drl_env.robot.robot import Robot
from typing import List

class Planner(ABC):

    def __init__(self, robot: Robot) -> None:
        super().__init__()

        self.robot: Robot = robot

    @abstractmethod
    def plan(self, q_goal) -> List:
        """
        This method will use the planner to plan a trajectory from the robot's current joint configuration to
        the goal.
        The result has to be a list of joint positions which for a valid, collision-free trajectory between the start and q_goal.
        """
        pass
