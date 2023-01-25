from world.world import World
import numpy as np
import pybullet as pyb
from world.obstacles.human import Human
from world.obstacles.pybullet_shapes import Box
from world.obstacles.shelf.shelf import ShelfObstacle
from random import choice

__all__ = [
    'KukaShelfExperiment'
]

class KukaShelfExperiment(World):
    """
    Implements the experiment world designed for the Kuka KR16 with two shelves and humans walking.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       shelves_positions: list,
                       shelves_rotations: list,
                       humans_positions: list,
                       humans_rotations: list,
                       humans_trajectories: list,
                       target_pos_override: list=[],
                       target_rot_override: list=[],
                       start_override: list=[]):
        super().__init__(workspace_boundaries, sim_step)

        # positions and rotations of the shelves as numpy arrays
        self.shelves_position = [np.array(position) for position in shelves_positions]
        self.shelves_rotations = [np.array(rotation) for rotation in shelves_rotations]

        # initial positions and rotations of the humans as numpy arrays
        self.humans_positions = [np.array(position) for position in humans_positions]
        self.humans_rotations = [np.array(rotation) for rotation in humans_rotations]
        # trajectories of the humans as numpy arrays
        self.humans_trajectories = [[np.array(position) for position in trajectory] for trajectory in humans_trajectories]

        # overrides for the target positions, useful for eval, a random one will be chosen
        self.target_pos_override = [np.array(position) for position in target_pos_override]
        self.target_rot_override = [np.array(rotation) for rotation in target_rot_override]
        self.start_override = [np.array(position) for position in start_override]

        # shelf params
        self.shelf_params = {
            "rows": 5,
            "cols": 5,
            "element_size": .5,
            "shelf_depth": .5,
            "wall_thickness": .01
        }

        # keep track of objects
        self.obstacle_objects = []

    def build(self):
        # ground plate
        self.objects_ids.append(pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01]))

        # build shelves
        for position, rotation in zip(self.shelves_position, self.shelves_rotations):
            shelf = ShelfObstacle(position, rotation, [], 0, self.shelf_params)
            self.obstacle_objects.append(shelf)
            self.objects_ids.append(shelf.build())
        
        # build humas
        for position, rotation, trajectory in zip(self.humans_positions, self.humans_rotations, self.humans_trajectories):
            human = Human(position, rotation, trajectory, self.sim_step)
            self.obstacle_objects.append(human)
            self.objects_ids.append(human.build())

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []
    
    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move()

    def create_ee_starting_points(self) -> list:
        if self.start_override:
            random_start = choice(self.start_override)
            return [(random_start, None)]
        else:
            pass  # TODO later

    def create_position_target(self) -> list:
        if self.target_pos_override:
            random_target = choice(self.target_pos_override)
            return [random_target]
        else:
            pass  # TODO later

    def create_rotation_target(self) -> list:
        if self.target_rot_override:
            random_rot = choice(self.target_rot_override)
            return [random_rot]
        else:
            pass  # TODO later