from modular_drl_env.world.world import World


class IsaacWorld(World):
    def perform_collision_check(self):
        self.collision = False

    def generate_gound_plane(self):
        pass