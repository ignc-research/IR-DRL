from world.obstacles.obstacle import Obstacle
import pybullet as pyb
import numpy as np
from typing import Union

class Sphere(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, velocity: float, radius: float, color: list=[0.75,0.75,0.75,1]) -> None:
        super().__init__(position, rotation, trajectory, velocity)

        self.radius = radius
        self.color = color

    def build(self) -> int:
        self.object_id = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.radius, rgbaColor=[0.75,0.75,0.75,1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=self.radius),
                                    basePosition=self.position_orig)
        return self.object_id

class Box(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, velocity: float, halfExtents: Union[list, np.ndarray], color=[0.5,0.5,0.5,1]) -> None:
        super().__init__(position, rotation, trajectory, velocity)

        self.color = color
        self.halfExtents = halfExtents

    def build(self) -> int:
        self.object_id = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=self.halfExtents, rgbaColor=[0.5,0.5,0.5,1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=self.halfExtents),
                                    basePosition=self.position_orig)

        return self.object_id