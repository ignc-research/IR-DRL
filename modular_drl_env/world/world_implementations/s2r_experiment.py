from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box, Sphere
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
import numpy as np
from random import choice, shuffle, sample
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'S2RExperiment'
]

class S2RExperiment(World):
    """
    This class replicates our real world setup and implements a few moderately constricted experiments
    with semi-random obstacles on it.
    Note: All of the below is written with the assumption that there is only one UR5 at [0, 0, 0.01] and [0, 0, -180] orientation.
          Anything else will probaly create bad results.
    """

    def __init__(self, 
                 sim_step: float, 
                 sim_steps_per_env_step: int, 
                 env_id: int, 
                 assets_path: str,
                 experiments: list=[],
                 max_num_obstacles: int=3):
        super().__init__([-2, 2, -2, 2, -1, 5], sim_step, sim_steps_per_env_step, env_id, assets_path)

        # experiments, empty list = all available ones
        self.experiments = [1] if not experiments else experiments

        # max number of obstacles that can appear in an experiment
        self.max_num_obstacles = max_num_obstacles

        # multiplier for pre-generation of obstacles
        self.generation_mult = 10

        # measurements for random obstacles
        self.box_low = np.array([0.025, 0.025, 0.025])
        self.box_high = np.array([0.055, 0.055, 0.055])
        self.sphere_low = 0.01
        self.sphere_high = 0.025

        # storage position
        self.storage_pos = np.array([0, 0, -10])

    def set_up(self):
        # ground plate
        self.ground_plate = GroundPlate()
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))  # move it lower because our floor is lower in this experiment
        # add the table the robot is standing on
        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5])
        self.table.build()

        # pre-generate necessary geometry
        for _ in range(self.max_num_obstacles * self.generation_mult):
            if np.random.random() > 0.3:  # 70% for box
                halfExtents = np.random.uniform(low=self.box_low, high=self.box_high, size=(3,))
                obst = Box(self.storage_pos, np.random.normal(size=(4,)), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents)
            else:  # 30% for sphere
                radius = np.random.uniform(low=self.sphere_low, high=self.sphere_high)
                obst = Sphere(self.storage_pos, [], self.sim_step, self.sim_steps_per_env_step, 0, radius=radius)
            obst.build()
            self.obstacle_objects.append(obst)

    def reset(self, success_rate: float):
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        # move currently used obstacles into storage
        for obst in self.active_objects[2:]:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.storage_pos + offset)
        # reset active objects
        self.active_objects = [self.ground_plate, self.table]

        # get number of obstacles for this run
        num_obsts = round(success_rate * self.max_num_obstacles)

        # pick one of the active experiments randomly
        experiment = choice(self.experiments)
        # and build it
        eval("self._build_exp" + str(experiment) + "(num_obsts)")

    def _build_exp1(self, num_obsts):
        import pybullet as pyb
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        # move the end effector in a straight line across the table, obstacles might appear on the line or close to it, 
        start_joints = np.array([-0.86, -1.984, 1.984, -1.653, -1.554, 0])  # cartesian 0.255 -0.385 0.367
        start_pos = np.array([0.302, -0.181, 0.357])
        end_pos = np.array([0.302, 0.45, 0.357])  # straight line across the table along the y axis
        diff = end_pos - start_pos

        while True:
            random_obsts = sample(self.obstacle_objects, num_obsts)

            for obst in random_obsts:
                waylength = np.random.uniform(low=0.3, high=0.85)
                random_pos = start_pos + waylength * diff
                if np.random.random() > 0.5:  # 50% of obstacles will be moving around
                    # generate random trajectory of random velocity and directions
                    random_vel = np.random.uniform(low=0.5, high=2)
                    traj = []
                    for _ in range(round(np.random.uniform(low=1, high=3))):
                        random_dir = np.random.uniform(low=-1, high=1, size=(3,))
                        random_dir = (random_dir * np.random.uniform(low=0.05, high=0.2)) / np.linalg.norm(random_dir)
                        traj.append(random_dir)
                    obst.move_step = random_vel * self.sim_step * self.sim_steps_per_env_step
                    obst.trajectory = traj
                else:
                    obst.velocity = 0
                    obst.trajectory = []
                obst.move_base(random_pos)

            # now move the robot to the start
            self.robots[0].moveto_joints(start_joints, False, self.robots[0].all_joints_ids)
            # check if the start scenario is valid
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            if pyb_u.collision:
                for obst in random_obsts:
                    obst.move_base(self.storage_pos)
                continue
            break
        # if we get here, all is good to go
        self.active_objects += random_obsts
        self.position_targets = [end_pos]
        # TODO: joint targets

    def update(self):
        for obstacle in self.active_objects:
            obstacle.move_traj()
    



