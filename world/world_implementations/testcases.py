from world.world import World
import numpy as np
import pybullet as pyb
from random import choice
from world.obstacles.pybullet_shapes import Box

__all__ = [
    'TestcasesWorld'
]

class TestcasesWorld(World):
    """
    Implements the testcases as created by Yifan.
    Note: this class assumes that the first robot mentioned in the config is the one doing the experiment!
    """

    def __init__(self, sim_step:float, env_id:int, test_mode: int):
        super().__init__([-0.4, 0.4, 0.3, 0.7, 0.2, 0.4], sim_step, env_id)

        self.test_mode = test_mode # 0: random, 1: one plate, 2: moving obstacle, 3: two plates
        self.current_test_mode = 0  # for random
        self.test3_phase = 0  # test3 has two phases

        # hardcoded end effector start positions, one per test case
        self.robot_ee_start_positions = [np.array([0.15, 0.4, 0.3]), np.array([0.1, 0.3, 0.33]), np.array([0.25, 0.4, 0.3])]
        self.robot_ee_start_orientations = [np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi]))]

        # hardcoded targets, per test case
        self.position_target_1 = np.array([-0.15, 0.4, 0.3])
        self.position_target_2 = np.array([-0.3, 0.45, 0.25])  # slightly changed from original
        self.position_target_3_1 = np.array([0, 0.4, 0.25])
        self.position_target_3_2 = np.array([-0.25 , 0.4, 0.25])

        # moving obstacle for test case 2
        self.moving_plate = None

        self.obstacle_objects = []

    def build(self):
        # add ground plate
        ground_plate = pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01])
        self.objects_ids.append(ground_plate)
        if self.current_test_mode == 1:
            self._build_test_1()
        elif self.current_test_mode == 2:
            self._build_test_2()
        elif self.current_test_mode == 3:
            self._build_test_3()
            
    def reset(self):
        if self.test_mode == 0:
            self.current_test_mode = choice([1, 2, 3])
        else:
            self.current_test_mode = self.test_mode
        self.objects_ids = []
        self.ee_starting_points = []
        self.moving_plate = None
        self.aux_object_ids = []
        self.test3_phase = 0
        self.obstacle_objects = []

    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move()
        if self.current_test_mode == 3:
            if self.test3_phase == 0:
                # this is only works if the first robot is the one performing the test as we require for this class
                dist_threshold = self.robots_in_world[0].goal.distance_threshold  # warning: this will crash if the goal has no such thing as a distance threshold
                ee_pos = self.robots[0].position_rotation_sensor.position
                dist = np.linalg.norm(ee_pos - self.position_target_3_1)
                if dist <= dist_threshold:
                    # overwrite current with new target
                    self.position_targets = [self.position_target_3_2]
                    for robot in self.robots_in_world[1:]:
                        self.position_targets.append([])
    
    def _build_test_1(self):
        obst = Box(np.array([0.0,0.4,0.3]), [0, 0, 0, 1], [], 0, [0.002,0.1,0.05])
        self.obstacle_objects.append(obst)
        self.objects_ids.append(obst.build())

    def _build_test_2(self):
        obst = Box([-0.3, 0.4, 0.3], [0, 0, 0, 1], [np.array([-0.3, 0.4, 0.3]), np.array([-0.3, 0.8, 0.3])], 0.15, [0.05,0.05,0.002])
        self.obstacle_objects.append(obst)
        self.objects_ids.append(obst.build())

    def _build_test_3(self):
        obst1 = Box([-0.1,0.4,0.26], [0, 0, 0, 1], [0.002,0.1,0.05])
        obst2 = Box([0.1,0.4,0.26], [0, 0, 0, 1], [0.002,0.1,0.05])
        self.obstacle_objects.append(obst1)
        self.obstacle_objects.append(obst2)
        self.objects_ids.append(obst1.build())
        self.objects_ids.append(obst2.build())

    def create_ee_starting_points(self) -> list:
        self.ee_starting_points.append((self.robot_ee_start_positions[self.current_test_mode - 1], self.robot_ee_start_orientations[self.current_test_mode - 1]))
        for robot in self.robots_in_world[1:]:
            self.ee_starting_points.append((None, None))
        return self.ee_starting_points

    def create_position_target(self):
        self.position_targets = []
        if self.current_test_mode == 1:
            self.position_targets.append(self.position_target_1)
        elif self.current_test_mode == 2:
            self.position_targets.append(self.position_target_2)
        elif self.current_test_mode == 3:
            self.position_targets.append(self.position_target_3_1)
        for robot in self.robots_in_world:
            self.position_targets.append([])
        return self.position_targets

    def create_rotation_target(self) -> list:
        return None  # not needed here for now