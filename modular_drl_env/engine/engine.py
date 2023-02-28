from abc import ABC, abstractmethod


class Engine(ABC):
    """
    Abstract base class that handles calls to physics engine methods in the main environment.py file.
    This does not include specific e.g. sensor or robot implementations, these are handled by their own subclasses.
    """

    def __init__(self, use_physics_sim: bool) -> None:
        super().__init__()
        self.use_physics_sim = use_physics_sim  # determines how objects are moved within the engine, either by brute setting their position or correct physics simulation

    @abstractmethod
    def initialize(self, display_mode: bool, sim_step: float, gravity: list, assets_path: str):
        """
        This method starts the engine in the python code.
        It should also set several attributes using the parameters:
        - display_mode: a bool that determines whether to render a GUI for the user (True) or not (False)
        - sim_step: a float number that determines the sim time that passes with call of Engine.step
        - gravity: a 3-vector that determines the gravitational force along the world xyz-axes
        - assets_path: a string containing the absolute path of the assets folder from where the engine will load in meshes
        """
        pass

    @abstractmethod
    def step(self):
        """
        This method should simply let sim time pass within the engine. This should not apply commands or forces to objects in the simulation on its own.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method should reset the entire simulation, meaning that all objects should be deleted and everything be reset.
        """
        pass
    