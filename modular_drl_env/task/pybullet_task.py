import pybullet as pyb
from modular_drl_env.task.task import Task
import numpy as np
import torch

from typing import List, Tuple, Union, Optional

from modular_drl_env.robot.robot import Robot
from modular_drl_env.world.obstacles.obstacle import Obstacle
from modular_drl_env.world.obstacles.shapes import *
from modular_drl_env.world.obstacles.urdf_object import *
from modular_drl_env.sensor import Sensor

class PybulletTask(Task):

    def __init__(self, asset_path: str, step_size: float, headless: bool = True) -> None:
        super().__init__(asset_path, step_size, headless)

        # start up pybullet
        disp = pyb.DIRECT if headless else pyb.GUI
        pyb.connect(disp)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)  # increases graphical quality in case we're rendering
        pyb.setAdditionalSearchPath(asset_path)

        # set some physics attributes
        pyb.setTimeStep(step_size)
        pyb.setGravity([0, 0, -9.8])

        # tracking
        self._robots = []
        self._geometry = []
        self._objects = []
        self._sensors = []

    def set_up(
            self, 
            robots: List[Robot],
            obstacles: List[Obstacle],
            sensors: List[Sensor],
            num_envs: int,
            boundaries: Tuple[float, float, float]
        ) -> Tuple[List[int], List[int], List [int]]:

        # spawn robots
        for i, robot in enumerate(robots):
            pyb_id = pyb.loadURDF(fileName=robot.urdf_path,
                                  basePosition=[0, 0, 0],  # TODO
                                  baseOrientation=[0, 0, 0, 1],  # TODO
                                  globalScaling=1,  # TODO
                                  useFixedBase=True)  # TODO 
            
            self._robots.append(pyb_id)
            self._objects.append(pyb_id)
        
        # spawn geometry
        for i, obstacle in enumerate(obstacles):
            if isinstance(obstacle, Box):
                create_id = self._create_box(obstacle.args)
            elif isinstance(obstacle, Sphere):
                create_id = self._create_sphere(obstacle.args)
            elif isinstance(obstacle, Cylinder):
                create_id = self._create_cylinder(obstacle.args)
            elif isinstance(obstacle, URDFObject):
                create_id = self._load_URDF()
            else:
                raise f"Obstacle {type(obstacle)} implemented"

            self._geometry.append(create_id)
            self._objects.append(create_id)

        for i, sensor in enumerate(sensors):
            raise "Sensors are not implemented"

        return list(range(len(self._geometry), len(self._objects))), list(range(len(self._geometry))), list(range(len(self._sensors)))
    
    def set_joint_positions(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
        for i, robot_index in enumerate(robot_indices):
            for joint_index in joint_indices:
                pyb.resetJointState(self._robots[robot_index], joint_index, positions[i][joint_index])

    def set_joint_position_targets(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
        for i, robot_index in enumerate(robot_indices):
            pyb.setJointMotorControlArray(self._robots[robot_index], 
                                          joint_indices, 
                                          controlMode=pyb.POSITION_CONTROL, 
                                          targetPositions=positions[i]) 
                                          #forces=forces.tolist())

    def set_joint_velocities(
            self,
            velocities: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:   
        for i, robot_index in enumerate(robot_indices):
            for joint_index in joint_indices:
                position = pyb.getJointState(self._robots[robot_index], joint_index)[0]
                pyb.resetJointState(self._robots[robot_index], joint_index, position, velocities[i][joint_index])

    def set_joint_velocity_targets(
            self,
            velocities: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:                                 

        for i, robot_index in enumerate(robot_indices):
            pyb.setJointMotorControlArray(self._robots[robot_index], 
                                          joint_indices, 
                                          controlMode=pyb.VELOCITY_CONTROL, 
                                          targetVelocities=velocities[i]) 
                                          #forces=forces.tolist())

    def set_local_poses(
            self,
            translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
        ) -> None:

        for i, object_index in enumerate(indices):
            pyb.resetBasePositionAndOrientation(self._objects[object_index], translations[i], orientations[i])

    def get_local_poses(
            self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None
        ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:

        pos = []
        ori = []
        for i, object_index in enumerate(indices):
            p, o = pyb.getBasePositionAndOrientation(self._objects[object_index])
            pos.append(p)
            ori.append(o)
        return np.array(pos), np.array(ori)
    
    def get_collisions(self) -> List[Tuple[int, int]]:
        return [(entry[1], entry[2]) for entry in pyb.getContactPoints()]

    def step(self):
        pyb.stepSimulation()
    


# TODO/Anmerkungen interface (?)
# 1. URDF position
# 2. URDF für Nicht-Roboter / Mesh laden
# 3. get/set local pose getrennt für Roboter und Objekte
# 4. in set_joint_position_targets wird joint_indices nicht benutzt
# 5. Obstacles: abstrakte build methode nutzen, die unter der Haube die korrekten Engine Befehle aufruft
# 6. num_robots sieht falsch aus?
# 7. Groundplane notwendig?
# 8. wie werden joint ids gemanaged?
# 9. position/velo targets: forces?
# 10. multiple Kollisionen
