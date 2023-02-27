from ..engine import Engine

class IsaacEngine(Engine):
    def __init__(self, use_physics_sim: bool) -> None:
        super().__init__(use_physics_sim)