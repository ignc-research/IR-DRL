from modular_drl_env.world.world import World
import numpy as np
from random import choice
from modular_drl_env.world.obstacles.shapes import Box
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

__all__ = [
    'TestcasesWorld'
]

class TestcasesWorld(World):
    """
    Implements the testcases as created by Yifan.
    Note: this class assumes that the first robot mentioned in the config is the one doing the experiment!
    """

    def __init__(self, sim_step:float, env_id:int, assets_path: str, test_mode: int):
        #super().__init__([-0.4, 0.4, 0.3, 0.7, 0.2, 0.4], sim_step, env_id)
        super().__init__([-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], sim_step, env_id, assets_path)

        self.test_mode = test_mode # 0: random, 1: one plate, 2: moving obstacle, 3: two plates
        self.current_test_mode = 0  # for random
        self.test3_phase = 0  # test3 has two phases

        # hardcoded end effector start positions, one per test case
        self.robot_start_joint_angles = [np.array([-2.05547714,  1.25192761, -1.95051253, -0.90225911, -1.56962013, -0.48620892]),
                                         np.array([-1.9669801,   1.22445893, -2.00302124, -0.82290244, -1.56965578, -0.3975389 ]),
                                         np.array([-2.15547714,  1.15192761, -1.85051253, -0.90225911, -1.56962013, -0.48620892])]

        # hardcoded targets, per test case
        self.position_target_1 = np.array([-0.15, 0.4, 0.3])
        self.position_target_2 = np.array([-0.3, 0.45, 0.25])  # slightly changed from original
        self.position_target_3_1 = np.array([0, 0.4, 0.25])
        self.position_target_3_2 = np.array([-0.25 , 0.4, 0.25])

        self.active_obstacles = []

        self.position_nowhere = np.array([0, 0, -10])

    def set_up(self):
        # add ground plate
        pyb_u.add_ground_plane(np.array([0, 0, -0.01]))
        
        # set up testcase geometry
        self.obst1 = Box(self.position_nowhere, [0, 0, 0, 1], [], 0, [0.002,0.1,0.05])
        self.obst1.build()
        self.obst2 = Box(self.position_nowhere, [0, 0, 0, 1], [np.array([0, 0, 0]), np.array([0, 0.4, 0])], 0.0015, [0.05,0.05,0.002])
        self.obst2.build()
        self.obst3a = Box(self.position_nowhere, [0, 0, 0, 1], [], 0, [0.002,0.1,0.05])
        self.obst3a.build()
        self.obst3b = Box(self.position_nowhere, [0, 0, 0, 1], [], 0, [0.002,0.1,0.05])
        self.obst3b.build()

        self.obstacle_objects.append(self.obst1)
        self.obstacle_objects.append(self.obst2)
        self.obstacle_objects.append(self.obst3a)
        self.obstacle_objects.append(self.obst3b)

    def reset(self, success_rate):
        if self.test_mode == 0:
            self.current_test_mode = choice([1, 2, 3])
        else:
            self.current_test_mode = self.test_mode

        self.test3_phase = 0


        for obst in self.active_obstacles:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.position_nowhere + offset)
        self.active_obstacles = []
        self.position_targets = []
        # first move everything into storage
        for obst in self.obstacle_objects:
            obst.move_base(self.position_nowhere)

        if self.current_test_mode == 1:
            self.obst1.move_base(np.array([0, 0.4, 0.3]))
            self.active_obstacles.append(self.obst1)
            self.position_targets = [self.position_target_1]
        elif self.current_test_mode == 2:
            self.obst2.move_base(np.array([-0.3, 0.4, 0.3]))
            self.active_obstacles.append(self.obst2)
            self.position_targets = [self.position_target_2]
        elif self.current_test_mode == 3:
            self.obst3a.move_base(np.array([-0.1, 0.4, 0.26]))
            self.obst3b.move_base(np.array([0.1, 0.4, 0.26]))
            self.active_obstacles.append(self.obst3a)
            self.active_obstacles.append(self.obst3b)
            self.position_targets = [self.position_target_3_1]

        self.robots[0].moveto_joints(self.robot_start_joint_angles[self.current_test_mode - 1], False)

    def update(self):
        for obstacle in self.active_obstacles:
            obstacle.move_traj()
        if self.current_test_mode == 3:
            if self.test3_phase == 0:
                # this is only works if the first robot is the one performing the test as we require for this class
                dist_threshold = self.robots[0].goal.distance_threshold  # warning: this will crash if the goal has no such thing as a distance threshold
                ee_pos = self.robots[0].position_rotation_sensor.position
                dist = np.linalg.norm(ee_pos - self.position_target_3_1)
                if dist <= dist_threshold * 1.5:
                    # overwrite current with new target
                    self.position_targets = [self.position_target_3_2]
                    for robot in self.robots[1:]:
                        self.position_targets.append([])
                    self.test3_phase = 1