from modular_drl_env.world.world import World
import numpy as np
from modular_drl_env.world.obstacles.human import Human
from modular_drl_env.world.obstacles.shapes import Box
from modular_drl_env.world.obstacles.shelf.shelf import ShelfObstacle
from random import choice, sample
from modular_drl_env.util.quaternion_util import rotate_vector
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

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
                       shelves: dict,
                       shuffle_humans: int,
                       humans: dict,
                       obstacles: dict,
                       shelf_params: dict={}):
        super().__init__(workspace_boundaries, sim_step, env_id)

        # overrides for the target positions, useful for eval, a random one will be chosen

        # get the obstacle setup
        self.shelf_list = shelves
        self.human_list = humans
        self.obstacle_list = obstacles

        # set shuffle mode
        # if shuffle is larger than 0, instead of spawning all entries in humans
        # we will spawn a random pick from the available positions
        self.shuffle_humans = shuffle_humans

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

        # keep track of objects
        self.shelves = []
        self.humans = []
        self.obstacle_objects = []

        # pre initialize the shelves to prevent their URDFs from being generated over and over
        for shelf_entry in self.shelf_list:
            shelf = ShelfObstacle(shelf_entry["position"], shelf_entry["rotation"], [], 0, self.env_id, self.shelf_params)
            self.shelves.append(shelf)

    def build(self, success_rate: float):
        # ground plate
        self.objects_ids.append(pyb_u.add_ground_plane(np.array([0, 0, -0.01])))

        # build shelves
        for shelf in self.shelves:
            self.objects_ids.append(shelf.build())
        
        # build humans
        if self.shuffle_humans == 0:
            humans_to_build = self.human_list
        else:
            num = choice(list(range(self.shuffle_humans))) + 1
            humans_to_build = sample(self.human_list, num)
        for entry in humans_to_build:
            human = Human(entry["position"], entry["rotation"], entry["trajectory"], self.sim_step * 2, 0.2)
            self.humans.append(human)
            self.objects_ids.append(human.build()) 

        # build obstacles
        for entry in self.obstacle_list:
            x_min, x_max, y_min, y_max, z_min, z_max = entry["zone"]
            zone_low = [x_min, y_min, z_min]
            zone_high = [x_max, y_max, z_max]
            v_min, v_max = entry["velocity"]
            b_min, b_max, l_min, l_max, h_min, h_max = entry["extents"]
            extent_low = [b_min, l_min, h_min]
            extent_high = [b_max, l_max, h_max]
            random_num = choice(list(range(entry["num"]))) + 1
            for i in range(random_num):
                v = np.random.uniform(low=v_min, high=v_max)
                pos = np.random.uniform(low=zone_low, high=zone_high, size=(3,))
                traj = []
                for i in range(3):
                    traj.append(np.random.uniform(low=zone_low, high=zone_high, size=(3,)))
                extents = np.random.uniform(low=extent_low, high=extent_high, size=(3,))
                box = Box(pos, np.array([0, 0, 0, 1]), traj, v * self.sim_step, extents)
                self.obstacle_objects.append(box)
                self.objects_ids.append(box.build())

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

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        for human in self.humans:
            del human
        self.humans = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []
        self.aux_object_ids = []
    
    def update(self):
        for obstacle in self.obstacle_objects:
            obstacle.move()
        for human in self.humans:
            human.move()