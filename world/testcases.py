from world.world import World
import numpy as np
import pybullet as pyb
from random import choice

class TestcasesWorld(World):
    """
    Implements the testcases as created by Yifan.
    """

    def __init__(self, test_mode: int):
        super().__init__([-0.4, 0.4, 0.3, 0.7, 0.2, 0.4], [], [])  # this class will run base position and orientation managing on its own

        self.test_mode = test_mode # 0: random, 1: one plate, 2: moving obstacle, 3: two plates
        self.current_test_mode = 0  # for random

        # base positions (hardcoded support for up to two robots)
        self.robot_base_positions = [np.array([0.0, -0.12, 0.5]), np.array([0.0, 1.12, 0.5])]
        self.robot_base_orientations = [np.array(pyb.getQuaternionFromEuler([0, 0, 0])), np.array(pyb.getQuaternionFromEuler([0, 0, np.pi]))]

        # hardcoded end effector start positions, one per test case
        self.robot_ee_start_positions = [[np.array([0.15, 0.4, 0.3]), np.array([0.1, 0.3, 0.33]), np.array([0.25, 0.4, 0.3])],
                                         [np.array([0.15, 0.6, 0.3]), np.array([0.1, 0.7, 0.33]), np.array([0.25, 0.6, 0.3])]]  # TODO: the ones for a potential roboter are temporary and have to be tested
        self.robot_ee_start_orientations = [[np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi]))],
                                            [np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi])), np.array(pyb.getQuaternionFromEuler([np.pi, 0, np.pi]))]]  # TODO: the ones for the second roboter are very likely nonsense

        # hardcoded targets, per test case
        self.position_targets_1 = [np.array([-0.15, 0.4, 0.3]), np.array([-0.15, 0.6, 0.3])]
        self.position_targets_2 = [np.array([-0.3, 0.45, 0.25]), np.array([-0.3, 0.55, 0.25])]  # slightly changed from original
        self.position_targets_3_1 = [np.array([0, 0.4, 0.25]), np.array([0, 0.6, 0.25])]
        self.position_targets_3_2 = [np.array([-0.25 , 0.4, 0.25]), np.array([-0.25, 0.4, 0.25])]

        # moving obstalce for test case 2
        self.moving_plate = None

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

    def update(self):
        if self.current_test_mode == 2:
            self.moving_plate_position[1] += 1 * 0.15 * 0.005
            pyb.resetBasePositionAndOrientation(self.moving_plate, self.moving_plate_position, [0, 0, 0, 1])

    
    def _build_test_1(self):
        obst = pyb.createMultiBody(baseMass=0,
                                   baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05], rgbaColor=[0.5,0.5,0.5,1]),
                                   baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
                                   basePosition=[0.0,0.4,0.3])

        self.objects_ids.append(obst)

    def _build_test_2(self):

        self.moving_plate = pyb.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.05,0.05,0.002], rgbaColor=[0.5,0.5,0.5,1]),
                        baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
                        basePosition=[-0.3, 0.4, 0.3]
                    )
        self.moving_plate_position = [-0.3, 0.4, 0.3]
        self.objects_ids.append(self.moving_plate)

    def _build_test_3(self):
        obst1 = pyb.createMultiBody(baseMass=0,
                                   baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05], rgbaColor=[0.5,0.5,0.5,1]),
                                   baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
                                   basePosition=[-0.1,0.4,0.26])

        obst2 = pyb.createMultiBody(baseMass=0,
                                   baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05], rgbaColor=[0.5,0.5,0.5,1]),
                                   baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[0.002,0.1,0.05]),
                                   basePosition=[0.1,0.4,0.26])
        self.objects_ids.append(obst1)
        self.objects_ids.append(obst2)

    def create_ee_starting_points(self) -> list:
        for idx, ele in enumerate(self.robot_ee_start_positions):
            self.ee_starting_points.append((self.robot_ee_start_positions[idx][self.current_test_mode-1], self.robot_ee_start_orientations[idx][self.current_test_mode-1]))
        return self.ee_starting_points

    def create_position_target(self):
        for idx, robot in enumerate(self.robots_in_world):
            if robot in self.robots_with_position:
                if self.current_test_mode == 1:
                    self.position_targets.append(self.position_targets_1[idx])
                elif self.current_test_mode == 2:
                    self.position_targets.append(self.position_targets_2[idx])
                elif self.current_test_mode == 3:
                    self.position_targets.append(self.position_targets_3_1[idx])
            else:
                self.position_targets.append([])
        return self.position_targets

    def create_rotation_target(self) -> list:
        return None  # not needed here for now

    def build_visual_aux(self):
        # create a visual border for the workspace
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])