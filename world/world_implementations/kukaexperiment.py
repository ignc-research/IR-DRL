from world.world import World
import numpy as np
import pybullet as pyb
from world.obstacles.human import Human
from world.obstacles.pybullet_shapes import Box
import pybullet_data as pyb_d
from random import choice

__all__ = [
    'KukaExperiment'
]

class KukaExperiment(World):
    """
    Implements the experiment world designed for the Kuka KR16 with two shelves and humans walking.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       shelves_positions: list,
                       shelves_rotations: list,
                       humans_positions: list,
                       humans_trajectories: list,
                       target_override: list):
        super().__init__(workspace_boundaries, sim_step)