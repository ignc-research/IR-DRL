from modular_drl_env.world.world import World
import numpy as np
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
from modular_drl_env.world.obstacles.shapes import Sphere, Box
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.util.quaternion_util import matrix_to_quaternion
from random import choice

class PlateExperiment(World):

    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int, assets_path: str, plate_dimensions: list):
        super().__init__(workspace_boundaries, sim_step, env_id, assets_path)
        # note: all the code in this class assumes that there is a single UR5 robot located at 0, 0, 0

        # storage position
        self.position_nowhere = np.array([0, 0, -10])

        # dimensions for randomly generated geometry, two entries: min & max
        self.plate_dimensions = plate_dimensions
        # width for plates, we set this to avoid problems with random generation
        self.plate_width = 0.0009

        # number of pre-generated variations for geometry
        self.num_pre_gen = 50

        # active objects in episode
        self.active_obstacles = []

    def set_up(self):
        # ground plate
        plate = GroundPlate()
        plate.build()

        for _ in range(self.num_pre_gen):
            random_dims = np.random.uniform(low=self.plate_dimensions[0], high=self.plate_dimensions[1], size=(2,))
            plate = Box(self.position_nowhere, [0, 0, 0, 1], [], 0, np.hstack([self.plate_width, random_dims]))
            plate.build()
            self.obstacle_objects.append(plate)

    def reset(self, success_rate: float):
        # reset attributes
        self.ee_starting_points = []
        for obst in self.active_obstacles:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.position_nowhere + offset)
        self.active_obstacles = []
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []

        while True:
            # generate starting point
            low = [-1.5, -1.5, 0.4]
            high = [1.5, 1.5, 1.3]
            while True:
                random_start = np.random.uniform(low=low, high=high, size=(3,))
                # check if the random point is in reach and not too close to the robot base
                if np.linalg.norm(random_start) < 0.6:
                    continue
                self.robots[0].moveto_xyz(random_start, False)
                position_start, rotation_start, _, _ = pyb_u.get_link_state(self.robots[0].object_id, "ee_link")
                joints_start, _ = pyb_u.get_joint_states(self.robots[0].object_id, self.robots[0].controlled_joints_ids)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if np.linalg.norm(position_start) > 1.5 or np.linalg.norm(position_start - random_start) > 5e-2 or pyb_u.collision:
                    continue
                break
            # now that we have a working start, we can generate a random goal
            while True:
                random_direction = np.random.uniform(low=[-1, -1, -0.1], high=[1, 1, 0.1], size=(3,)) 
                random_length = np.random.uniform(low=0.15, high=0.65)
                random_goal = random_start + (random_direction * random_length / np.linalg.norm(random_direction))
                if np.linalg.norm(random_goal) < 0.6 or np.linalg.norm(random_goal - random_start) < 0.3:
                    continue
                self.robots[0].moveto_xyz(random_goal, False)
                position_goal, rotation_goal, _, _ = pyb_u.get_link_state(self.robots[0].object_id, "ee_link")
                joints_goal, _ = pyb_u.get_joint_states(self.robots[0].object_id, self.robots[0].controlled_joints_ids)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if np.linalg.norm(position_goal) > 1.5 or np.linalg.norm(position_goal - random_goal) > 5e-2 or pyb_u.collision:
                    continue
                break
            # finally, we can place some object between the two and rotate it such that its largest face (in case of a plate) is oriented towards the start/goal
            tries = 0
            while True:
                tries += 1
                if tries > 500:
                    break
                random_obst = choice(self.obstacle_objects)
                # we randomly pick a spot along the way between goal and start
                random_mod = np.random.uniform(low=0.35, high=0.65)
                obst_pos = random_start + random_direction * random_mod * random_length / np.linalg.norm(random_direction)
                # we created all the plates in 0,0,0,1 rotation and the thin dimension along the x axis
                # we can use the random direction for this
                temp_vec = np.random.uniform(low=-1, high=1, size=(3,))
                temp_vec = temp_vec / np.linalg.norm(temp_vec)
                b = np.cross(random_direction / np.linalg.norm(random_direction), temp_vec)
                b = b / np.linalg.norm(b)
                c = np.cross(random_direction / np.linalg.norm(random_direction),b)
                c = c / np.linalg.norm(b)
                rot_mat = np.eye(3)
                rot_mat[:3, 0] = random_direction / np.linalg.norm(random_direction)
                rot_mat[:3, 1] = b
                rot_mat[:3, 2] = c
                rot_quat = matrix_to_quaternion(rot_mat)
                # now we can move the obstacle
                pyb_u.set_base_pos_and_ori(random_obst.object_id, obst_pos, rot_quat)
                # now check if there is any collision
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    pyb_u.set_base_pos_and_ori(random_obst.object_id, self.position_nowhere, np.array([0, 0, 0, 1]))
                    continue
                # check for starting position
                self.robots[0].moveto_joints(joints_start, False)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    pyb_u.set_base_pos_and_ori(random_obst.object_id, self.position_nowhere, np.array([0, 0, 0, 1]))
                    self.robots[0].moveto_joints(joints_goal, False)
                    continue
                break
            if tries <= 500:
                break
        # now we can set all the attributes
        self.active_obstacles.append(random_obst)
        #print(self.active_obstacles)
        self.ee_starting_points.append((random_start, rotation_start, joints_start))
        self.position_targets.append(random_goal)
        self.rotation_targets.append(rotation_goal)
        self.joints_targets.append(joints_goal)
        self.robots[0].moveto_joints(joints_start, False)

    def update(self):
        pass
        






