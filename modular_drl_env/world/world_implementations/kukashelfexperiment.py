from modular_drl_env.world.world import World
import numpy as np
from modular_drl_env.world.obstacles.human import Human
from modular_drl_env.world.obstacles.shapes import Box
from modular_drl_env.world.obstacles.urdf_object import URDFObject
from random import choice, sample
from modular_drl_env.util.quaternion_util import rotate_vector
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.shared.shelf_generator import ShelfGenerator

__all__ = [
    'KukaShelfExperiment'
]

class KukaShelfExperiment(World):
    """
    Implements the experiment world designed for the Kuka KR16 with two shelves and humans walking.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       env_id: int,
                       assets_path: str,
                       shelves: dict,
                       shelf_params: dict={}):
        super().__init__(workspace_boundaries, sim_step, env_id, assets_path)

        # get the obstacle setup
        self.shelf_list = shelves

        # shelf params
        if not shelf_params:
            self.shelf_params = {
                "rows": 5,  #x
                "cols": 5,  #+y
                "element_size": .5,
                "shelf_depth": .5, # +z
                "wall_thickness": .01
            }
        else:
            self.shelf_params = shelf_params


    def set_up(self):
        # check if the goal we're working with uses only position targets
        for robot in self.robots:
            if robot.goal.needs_a_rotation or robot.goal.needs_a_joint_position:
                raise Exception("Shelf Experiment does not support rotation or joint position goals.")

        # ground plane
        self.objects_ids.append(pyb_u.add_ground_plane(np.array([0, 0, -0.01])))

        # generate an appropriate URDF
        generator = ShelfGenerator(self.shelf_params)
        urdf_name = self.assets_path + "/runtime/shelf_" + str(self.env_id) + ".urdf"
        with open(urdf_name, "w") as outfile:
            outfile.write(generator.generate())
        
        # instantiante as obstacle as many times as needed
        for shelf_entry in self.shelf_list:
            shelf = URDFObject(shelf_entry["position"], shelf_entry["rotation"], [], 0, urdf_name)
            self.obstacle_objects.append(shelf)
            self.objects_ids.append(shelf.build())

        # TODO: add obstacles and humans again

    def reset(self, success_rate: float):

        # starting points
        for robot in self.robots:
            robot.moveto_joints(robot.resting_pose_angles, False)
        # create targets
        targets = []
        taken_shelf_pos = []
        for robot in self.robots:
            shelf = choice(self.shelf_list)
            shelf_pos = shelf["position"]
            shelf_rot = shelf["rotation"]
            # get random shelf drawer
            val = False
            while not val:
                col = choice(list(range(self.shelf_params["cols"])))
                row = choice(list(range(self.shelf_params["rows"])))
                if (col, row) not in taken_shelf_pos:
                    val = True
            taken_shelf_pos.append((col, row))
            # calculate local x y z coordinate
            z = self.shelf_params["shelf_depth"] / 2  # in the middle of the free space
            x = self.shelf_params["wall_thickness"] + self.shelf_params["element_size"] / 2 + col * (self.shelf_params["wall_thickness"] + self.shelf_params["element_size"])
            y = self.shelf_params["wall_thickness"] + self.shelf_params["element_size"] / 2 + row * (self.shelf_params["wall_thickness"] + self.shelf_params["element_size"])
            local_target = np.array([x, y, z])
            target = rotate_vector(local_target, shelf_rot) + shelf_pos
            targets.append(target)
        self.position_targets = targets       
    
    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move()