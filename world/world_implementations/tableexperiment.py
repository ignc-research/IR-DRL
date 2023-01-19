from world.world import World
import numpy as np
import pybullet as pyb
from world.obstacles.human import Human
import pybullet_data as pyb_d

__all__ = [
    'TableExperiment'
]

class TableExperiment(World):
    """
    Implements the table experiment with humans and moving obstacles by Kolja and Kai.
    """

    def __init__(self, workspace_boundaries: list, 
                       robot_base_positions: list, 
                       robot_base_orientations: list,
                       sim_step: float,
                       num_obstacles: int,
                       obstacle_velocities: list,
                       num_humans: int,
                       human_positions: list,
                       human_rotations: list,
                       human_trajectories: list,
                       ee_start_overwrite: list=[],
                       target_overwrite: list=[]):
        super().__init__(workspace_boundaries, robot_base_positions, robot_base_orientations, sim_step)
        # INFO: if multiple robot base positions are given, we will assume that the first one is the main one for the experiment

        self.num_obstacles = num_obstacles
        self.num_humans = num_humans
        self.obstacle_velocities = obstacle_velocities
        self.ee_start_overwrite = ee_start_overwrite
        self.target_overwrite = target_overwrite

        self.humans = []
        self.human_positions = human_positions
        self.human_rotations = human_rotations
        self.human_trajectories = human_trajectories
        
    def build(self):
        # ground plate
        self.objects_ids.append(pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01]))
        # table
        self.objects_ids.append(pyb.loadURDF(pyb_d.getDataPath()+"/table/table.urdf", useFixedBase=True, globalScaling=2))
        for i in range(self.num_humans):
            human = Human(self.human_positions[i], self.human_rotations[i], self.human_trajectories[i], self.sim_step, 1.5)
            human.build()
            self.humans.append(human)

    def reset(self):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for human in self.humans:
            del human
        self.humans = []

    def update(self):
        for human in self.humans:
            human.move()

    def create_ee_starting_points(self) -> list:
        # use the preset starting points if there are some
        if self.ee_start_overwrite:
            ret = []
            for idx in range(len(self.ee_start_overwrite)):
                standard_rot = np.array([np.pi, 0, np.pi])
                random_rot = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                standard_rot += random_rot * 0.1
                ret.append((self.ee_start_overwrite[idx], np.array(pyb.getQuaternionFromEuler(standard_rot.tolist()))))
                #ret.append((self.ee_start_overwrite[idx], None))
            return ret
        # otherwise generate randomly
        else:
            pass  # TODO

    def create_position_target(self) -> list:
        # use the preset targets if there are some
        if self.target_overwrite:
            self.position_targets = self.target_overwrite
            return self.target_overwrite
        # otherwise generate randomly
        else:
            pass  # TODO

    def create_rotation_target(self) -> list:
        return None  # not needed for now