from world.world import World
import numpy as np
import pybullet as pyb
from world.obstacles.human import Human
from world.obstacles.pybullet_shapes import Box
import pybullet_data as pyb_d
from random import choice

__all__ = [
    'TableExperiment'
]

class TableExperiment(World):
    """
    Implements the table experiment with humans and moving obstacles by Kolja and Kai.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       num_obstacles: int,
                       obstacle_velocities: list,
                       num_humans: int,
                       human_positions: list,
                       human_rotations: list,
                       human_trajectories: list,
                       human_reactive: list,
                       ee_starts: list=[],
                       targets: list=[],
                       obstacle_positions: list=[],
                       obstacle_trajectories: list=[]):
        super().__init__(workspace_boundaries, sim_step)
        # INFO: if multiple robot base positions are given, we will assume that the first one is the main one for the experiment

        self.num_obstacles = num_obstacles
        self.num_humans = num_humans
        self.obstacle_velocities = obstacle_velocities

        # all of the following lists serve as overwrites for env functionality
        # useful for getting a repeatable starting point for evaluation
        # if left as empty lists the env will generate random ones, useful for training
        self.ee_starts = ee_starts
        self.targets = targets
        self.obstacle_positions = obstacle_positions
        self.obstacle_trajectories = obstacle_trajectories

        # handle human stuff
        self.humans = []
        self.human_positions = human_positions
        self.human_rotations = human_rotations
        self.human_trajectories = human_trajectories
        self.human_reactive = human_reactive  # list of bools that determines if the human in question will raise his arm if the robot gets near enough
        self.human_ee_was_near = [False for i in range(num_humans)]  # see update method
        self.near_threshold = 0.5

        self.obstacle_objects = []
        
    def build(self):
        # ground plate
        self.objects_ids.append(pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01]))
        # table
        self.objects_ids.append(pyb.loadURDF(pyb_d.getDataPath()+"/table/table.urdf", useFixedBase=True, globalScaling=1.75))
        # humans
        for i in range(self.num_humans):
            human = Human(self.human_positions[i], self.human_rotations[i], self.human_trajectories[i], self.sim_step, 1.5)
            human.build()
            self.humans.append(human)
        # obstacles
        for i in range(self.num_obstacles):
            # if there are no given obstacle positions, randomly generate some
            if not self.obstacle_positions:
                pass
            else:
                position = self.obstacle_positions[i]
            # if there are no given obstacle trajectories, randomly generate some
            if not self.obstacle_trajectories:
                pass
            else:
                trajectory = self.obstacle_trajectories[i]
            # create random dimensions for obstacles 
            halfExtents = [0.05, 0.05, 0.05]
            # create random velocities / move steps
            move_step = 0
            # create obstacles
            obs = Box(position, [0, 0, 0, 1], trajectory, move_step, halfExtents, color=[1, 0, 0, 1])      
            self.objects_ids.append(obs.build())
            self.obstacle_objects.append(obs)    

    def reset(self):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for human in self.humans:
            del human
        self.humans = []

    def update(self):
        for idx, human in enumerate(self.humans):
            human.move()
            if self.human_reactive[idx]:
                near = False or self.human_ee_was_near[idx]
                if not near:  # check if end effector is near
                    for robot in self.robots_in_world:
                        if np.linalg.norm(robot.position_rotation_sensor.position - human.position) <= self.near_threshold:
                            near = True
                            self.human_ee_was_near[idx] = True
                            break
                if near:
                    human.raise_hands()

    def create_ee_starting_points(self) -> list:
        # use the preset starting points if there are some
        if self.ee_starts:
            ret = []
            for idx in range(len(self.ee_starts)):
                standard_rot = np.array([np.pi, 0, np.pi])
                random_rot = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                standard_rot += random_rot * 0.1
                ret.append((self.ee_starts[idx], np.array(pyb.getQuaternionFromEuler(standard_rot.tolist()))))
                #ret.append((self.ee_start_overwrite[idx], None))
            return ret
        # otherwise generate randomly
        else:
            pass  # TODO

    def create_position_target(self) -> list:
        # use the preset targets if there are some
        if self.targets:
            self.position_targets = self.targets
            return self.targets
        # otherwise generate randomly
        else:
            # as this is a predetermined experiment we chose goals from a curated list
            possible_targets = [np.array([-0.65, -0.65, 1.2]),
                                np.array([])
                                ]
            target = choice(possible_targets)
            self.position_targets = [target]
            return [target]

    def create_rotation_target(self) -> list:
        return None  # not needed for now