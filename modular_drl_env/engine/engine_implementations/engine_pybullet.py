import pybullet as pyb
from ..engine import Engine
from typing import List, TYPE_CHECKING

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

class PybulletEngine(Engine):

    def __init__(self, use_physics_sim: bool) -> None:
        super().__init__(use_physics_sim)

    def initialize(self, display_mode: bool, sim_step: float, gravity: list, assets_path: str):
        disp = pyb.DIRECT if not display_mode else pyb.GUI
        pyb.connect(disp)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
        pyb.setTimeStep(sim_step)
        pyb.setGravity(*gravity)
        pyb.setAdditionalSearchPath(assets_path)

    def step(self):
        if self.use_physics_sim:
            pyb.stepSimulation()
        else:
            pass

    def reset(self):
        pyb.resetSimulation()

    def perform_collision_check(self, robots: List["Robot"], obstacles: List["Obstacle"]) -> bool:
        pyb.performCollisionDetection()
        col = False
        # check for each robot with every obstacle
        for robot in robots:
            for obj in obstacles:
                if len(pyb.getContactPoints(robot.object_id, obj)) > 0:
                    col = True 
                    break
            if col:
                break  # this is to immediately break out of the outer loop too once a collision has been found
        # check for each robot with every other one
        if not col:  # skip if another collision was already detected
            for idx, robot in enumerate(robots[:-1]):
                for other_robot in robots[idx+1:]:
                    if len(pyb.getContactPoints(robot.object_id, other_robot.object_id)) > 0:
                        col = True
                        break
                if col:
                    break  # same as above
        return col
    
    def add_ground_plane(self):
        return pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01])
    
    def addUserDebugLine(self, lineFromXYZ: List[float], lineToXYZ: List[float]):
        return pyb.addUserDebugLine(lineFromXYZ, lineToXYZ)
