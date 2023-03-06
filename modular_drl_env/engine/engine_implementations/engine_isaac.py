from ..engine import Engine
import atexit

# Try importing all Issac modules in a try/except to allow compilation without it
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    pass


class IsaacEngine(Engine):
    def __init__(self, use_physics_sim: bool) -> None:
        super().__init__(use_physics_sim)

    def initialize(self, display_mode: bool, sim_step: float, gravity: list, assets_path: str):
        # setup simulation
        self.simulation = SimulationApp({"headless": not display_mode})

        # terminate simulation once program exits
        atexit.register(self.simulation.close)

        # todo: include sim_step, gravity, asset_path?
        print(assets_path)

    def step(self):
        # simulate physics step if pyhsics is enabled
        if self.use_physics_sim:
            self.simulation.update()

    def reset(self):
        raise "NotImplemented!"

    def add_ground_plane(self):
        raise "NotImplemented!"
