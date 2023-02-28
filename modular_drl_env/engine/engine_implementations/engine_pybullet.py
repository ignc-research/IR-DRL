import pybullet as pyb
from ..engine import Engine


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
