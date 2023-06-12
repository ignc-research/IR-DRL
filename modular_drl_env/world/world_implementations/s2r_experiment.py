from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box, Sphere
from modular_drl_env.world.obstacles.ground_plate import GroundPlate
import numpy as np
import pickle as pkl
from random import choice, shuffle, sample
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet as pyb

__all__ = [
    'S2RExperiment',
    'S2RExperimentVoxels'
]

class S2RExperiment(World):
    """
    This class replicates our real world setup and implements a few moderately constricted experiments
    with semi-random obstacles on it.
    Note: All of the below is written with the assumption that there is only one UR5 at [0, 0, 0.01] and [0, 0, -180] orientation.
          Anything else will probaly create bad results.
    """

    def __init__(self, 
                 sim_step: float, 
                 sim_steps_per_env_step: int, 
                 env_id: int, 
                 assets_path: str,
                 experiments: list=[],
                 experiments_weights: list=[],
                 max_num_obstacles: int=3):
        super().__init__([-2, 2, -2, 2, -1, 5], sim_step, sim_steps_per_env_step, env_id, assets_path)

        # experiments, empty list = all available ones
        self.experiments = [0, 1, 2] if not experiments else experiments
        self.experiments_weights = [1/len(self.experiments) for _ in self.experiments] if not experiments_weights else experiments_weights

        # max number of obstacles that can appear in an experiment
        self.max_num_obstacles = max_num_obstacles

        # multiplier for pre-generation of obstacles
        self.generation_mult = 10

        # measurements for random obstacles
        self.box_low = np.array([0.025, 0.025, 0.025])
        self.box_high = np.array([0.055, 0.055, 0.055])
        self.sphere_low = 0.01
        self.sphere_high = 0.025

        # storage position
        self.storage_pos = np.array([0, 0, -10])

        # storage array for specific geometry with defined sizes
        self.specific_obstacles = []

    def set_up(self):
        # ground plate
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))  # move it lower because our floor is lower in this experiment
        # add the table the robot is standing on
        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=False)
        self.table.build()

        # pre-generate necessary geometry
        #   stuff with specific dims
        halfExtents = [0.1, 0.3, 0.1]
        lengthy_box = Box(self.storage_pos, np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents, [0.75, 0.2, 0.1, 1])
        lengthy_box.build()
        halfExtents = [0.6, 0.1, 0.4]
        barrier_box_1 = Box(self.storage_pos, np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents, [0.2, 0.1, 0.75, 1])
        barrier_box_1.build()
        halfExtents = [0.1, 0.6, 0.4]
        barrier_box_2 = Box(self.storage_pos, np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents, [0.2, 0.1, 0.75, 1])
        barrier_box_2.build()
        halfExtents = [0.115, 0.0009, 0.115]
        somewhat_large_plate = Box(self.storage_pos, np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents, [0.294, 0, 0.51, 1])
        somewhat_large_plate.build()
        
        self.specific_obstacles += [self.ground_plate, self.table, lengthy_box, barrier_box_1, barrier_box_2, somewhat_large_plate]
        #   random shapes
        for _ in range(self.max_num_obstacles * self.generation_mult):
            if np.random.random() > 0.3:  # 70% for box
                halfExtents = np.random.uniform(low=self.box_low, high=self.box_high, size=(3,))
                obst = Box(self.storage_pos, np.random.normal(size=(4,)), [], self.sim_step, self.sim_steps_per_env_step, 0, halfExtents)
            else:  # 30% for sphere
                radius = np.random.uniform(low=self.sphere_low, high=self.sphere_high)
                obst = Sphere(self.storage_pos, [], self.sim_step, self.sim_steps_per_env_step, 0, radius=radius)
            obst.build()
            self.obstacle_objects.append(obst)

    def reset(self, success_rate: float):
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []
        # move currently used obstacles into storage
        for obst in self.active_objects[2:]:
            offset = np.random.uniform(low=-5, high=5, size=(3,))
            obst.move_base(self.storage_pos + offset)
        # reset active objects
        self.active_objects = [self.ground_plate, self.table]

        # get number of obstacles for this run
        num_obsts = round(success_rate * self.max_num_obstacles)

        # pick one of the active experiments randomly
        experiment = np.random.choice(self.experiments, p=self.experiments_weights)
        # and build it
        eval("self._set_up_exp" + str(experiment) + "(num_obsts)")

    def _set_up_exp0(self, num_obsts):
        # move the end effector in a straight line across the table, obstacles might appear on the line or close to it, 
        while True:
            random_x = np.random.uniform(low=0.25, high=0.5)
            random_z = np.random.uniform(low=0.25, high=0.5)
            random_start_end = choice([(0.05, 0.55), (0.55, 0.05)])
            #random_start_end = choice([(0.45, -0.181)])
            start_pos = np.array([random_x, random_start_end[0], random_z])
            end_pos = np.array([random_x, random_start_end[1], random_z])  # straight line across the table along the y axis
            diff = end_pos - start_pos
            random_obsts = sample(self.obstacle_objects, num_obsts)


            for obst in random_obsts:
                waylength = np.random.uniform(low=0.3, high=0.85)
                random_pos = start_pos + waylength * diff
                if np.random.random() > 0.5:  # 50% of obstacles will be moving around
                    # generate random trajectory of random velocity and directions
                    random_vel = np.random.uniform(low=0.5, high=2)
                    traj = []
                    for _ in range(round(np.random.uniform(low=1, high=3))):
                        random_dir = np.random.uniform(low=-1, high=1, size=(3,))
                        random_dir = (random_dir * np.random.uniform(low=0.05, high=0.2)) / np.linalg.norm(random_dir)
                        traj.append(random_dir)
                    obst.move_step = random_vel * self.sim_step * self.sim_steps_per_env_step
                    obst.trajectory = traj
                else:
                    obst.move_step = 0
                    obst.trajectory = []
                obst.move_base(random_pos)

            # now move the robot to the start (we move trhough the start angles first because the result of movetoxyz is path dependent)
            self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)
            self.robots[0].moveto_xyz(start_pos, False)
            # check if the start scenario is valid
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            if pyb_u.collision:
                for obst in random_obsts:
                    obst.move_base(self.storage_pos)
                continue
            break
        # if we get here, all is good to go
        self.active_objects += random_obsts
        # with 70% chance set up barriers to prevent the robot from abusing the free space behind it
        if np.random.random() > 0.3: 
            # set up the barrier boxes
            self.specific_obstacles[3].move_base(np.array([0.3, -0.55, 0.25]))
            self.specific_obstacles[4].move_base(np.array([-0.55, 0.3, 0.25]))
            self.active_objects += [self.specific_obstacles[3], self.specific_obstacles[4]]
        self.position_targets = [end_pos]
        # TODO: joint targets

    def _set_up_exp1(self, num_obsts):
        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)
        obst = self.specific_obstacles[2]
        random_y_start = np.random.uniform(low=0, high=0.3)
        random_z_start = np.random.uniform(low=0.25, high=0.4)
        obst.move_base(np.array([0.4, random_y_start, random_z_start]))
        obst.move_step = 0.35 * self.sim_step * self.sim_steps_per_env_step
        obst.trajectory = [np.array([0, 0.4 - random_y_start, 0.0]), np.array([0, -random_y_start, 0])]
        self.active_objects += [obst]
        targets = [np.array([0.55, 0.0, random_z_start]), np.array([0.4, -0.05, random_z_start]), np.array([0.4, 0.41, random_z_start])]
        self.position_targets = [choice(targets)]

    def _set_up_exp2(self, num_obsts):
        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)
        obst = self.specific_obstacles[2]
        random_y_start = np.random.uniform(low=0, high=0.3)
        random_z_start = np.random.uniform(low=0.1, high=0.4)
        obst.move_base(np.array([0.4, random_y_start, random_z_start]))
        obst.move_step = 0.35 * self.sim_step * self.sim_steps_per_env_step
        obst.trajectory = [np.array([0, 0, 0.4 - random_z_start]), np.array([0, 0, -random_z_start])]
        self.active_objects += [obst]
        targets = [np.array([0.4, random_y_start, 0.25]), np.array([0.4, random_y_start, 0.45]), np.array([0.55, random_y_start, 0.25]), np.array([0.55, random_y_start, 0.45])]
        self.position_targets = [choice(targets)]

    def _set_up_exp3(self, num_obsts):
        # random target without obstacles, sometimes the barriers appear to constrict the target onto the table
        if np.random.random() < 0.35: 
            # set up the barrier boxes
            self.specific_obstacles[3].move_base(np.array([0.3, -0.55, 0.25]))
            self.specific_obstacles[4].move_base(np.array([-0.55, 0.3, 0.25]))
            self.active_objects += [self.specific_obstacles[3], self.specific_obstacles[4]]
        # val = False
        # while not val:
        #     val = self._create_ee_starting_points(self.robots,)
        #     if not val:
        #         continue
        #     val = self._create_position_and_rotation_targets(self.robots)
        # self.robots[0].moveto_joints(self.ee_starting_points[0][2], False)
        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)
        self.position_targets = [np.random.uniform(low=[0, 0, 0.15], high=[0.7, 0.7, 0.7], size=(3,))]

    def _set_up_exp4(self, num_obsts):
        pyb_u.toggle_rendering(True)
        # move in a random line with a somewhat obstructive plate somewhere in between
        # the plate moves in rare cases

        # set up the barrier boxes, this prevents the agent from learning that it can simply perform a wide swing
        self.specific_obstacles[3].move_base(np.array([0.3, -0.55, 0.25]))
        self.specific_obstacles[4].move_base(np.array([-0.55, 0.3, 0.25]))
        self.active_objects += [self.specific_obstacles[3], self.specific_obstacles[4]]

        # loop, exits once a valid starting setup is found
        while True:
            low = [0, 0, 0.1]
            high = [0.8, 0.8, 0.6]
            # find a good starting point
            while True:
                random_start = np.random.uniform(low=low, high=high, size=(3,))
                if np.linalg.norm(random_start) < 0.25 or np.linalg.norm(random_start) > 1.5:
                    continue  # distance to robot base is too small or too far away
                self.robots[0].moveto_xyz(random_start, False)
                joints_start, _ = pyb_u.get_joint_states(self.robots[0].object_id, self.robots[0].controlled_joints_ids)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    continue  # point is in collision
                break  # exit this loop once a good point is found
            # now generate a random goal
            while True:
                random_direction = np.random.uniform(low=[-1, -1, -0.3], high=[1, 1, 0.3], size=(3,)) 
                random_length = np.random.uniform(low=0.15, high=0.65)
                random_goal = random_start + (random_direction * random_length / np.linalg.norm(random_direction))
                if np.linalg.norm(random_goal) < 0.25 or np.linalg.norm(random_goal - random_start) < 0.3:
                    continue  # too close to either robot base or the starting point
                self.robots[0].moveto_xyz(random_goal, False)
                position_goal, _, _, _ = pyb_u.get_link_state(self.robots[0].object_id, "ee_link")
                joints_goal, _ = pyb_u.get_joint_states(self.robots[0].object_id, self.robots[0].controlled_joints_ids)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if np.linalg.norm(position_goal) > 1.5 or np.linalg.norm(position_goal - random_goal) > 5e-2 or pyb_u.collision:
                    continue
                break
            # now we can set a plate between the two points
            # now loop and search for a position until it doesn't collide with the robot
            plate = self.specific_obstacles[5]
            tries = 0
            while True:
                # the whole setup might be impossible, so we have a counter to cutoff further tries at some point
                tries += 1
                if tries > 100:
                    plate.move_base(self.storage_pos, np.array([0, 0, 0, 1]))
                    break
                # we randomly pick a spot along the way between goal and start
                random_mod = np.random.uniform(low=0.35, high=0.65)
                obst_pos = random_start + random_direction * random_mod * random_length / np.linalg.norm(random_direction)
                plate.move_base(obst_pos)
                # now check for collision, the robot is still in the goal position
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    plate.move_base(self.storage_pos, np.array([0, 0, 0, 1]))
                    continue
                # check for starting position
                self.robots[0].moveto_joints(joints_start, False)
                pyb_u.perform_collision_check()
                pyb_u.get_collisions()
                if pyb_u.collision:
                    plate.move_base(self.storage_pos, np.array([0, 0, 0, 1]))
                    self.robots[0].moveto_joints(joints_goal, False)
                    continue
                break  # if we reach this line, both positions are fine and the plate is in a valid position
            if tries <= 100:
                break

        self.active_objects += [plate]
        self.position_targets = [position_goal]

    def update(self):
        for obstacle in self.active_objects:
            obstacle.move_traj()
    
class S2RExperimentVoxels(World):

    def __init__(self, 
                 voxel_frames_path: str,
                 recording_time: float,
                 sim_step: float, 
                 sim_steps_per_env_step: int, 
                 env_id: int, 
                 assets_path: str,
                 experiments: list=[],
                 experiments_weights: list=[],
                 voxel_size: float=0.035,
                 num_voxels: int=2000,
                 max_recording_sample_time: float=3,
                 min_recording_sample_time: float=0.5):
        """
        Same as the class above, just that we use pre-recorded voxels as obstacles.
        """
        
        super().__init__([-2, 2, -2, 2, -1, 5], sim_step, sim_steps_per_env_step, env_id, assets_path)

        # read frames for voxel movements
        with open(voxel_frames_path, "rb") as infile:
            self.voxel_frames = pkl.load(infile)
        self.voxels = []
        # properties for the frames
        self.frame_time = (recording_time / len(self.voxel_frames)) / 18  # time per frame
        self.max_num_frames = max_recording_sample_time / self.frame_time
        self.min_num_frames = min_recording_sample_time / self.frame_time
        self.current_frame = None
        self.start_frame = None
        self.end_frame = None

        # track time to correctly sync sim time with the frames
        self.time = 0
        
        # number of voxels used to display frames
        self.num_voxels = num_voxels
        # size of each voxel
        self.voxel_size = voxel_size

        # experiments,
        # 0: fully random frame, fully random start, fully random target
        # 1: curated scenarios with curated targets
        self.experiments = experiments
        self.experiments_weights = experiments_weights  # probabilities for reset for each experiment

        # storage location for voxels we don't use
        self.storage = np.array([0, 0, -10])
    
    def set_up(self):
        # ground plate
        self.ground_plate = GroundPlate(False)
        self.ground_plate.build()
        self.ground_plate.move_base(np.array([0, 0, -1]))  # move it lower because our floor is lower in this experiment
        # add the table the robot is standing on
        self.table = Box(np.array([0.25, 0.25, -0.5]), np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [0.3, 0.3, 0.5], seen_by_obstacle_sensor=False)
        self.table.build()
        self.obstacle_objects += [self.ground_plate, self.table]

        # create all the voxels 
        for _ in range(self.num_voxels):
            voxel = Box(self.storage, np.array([0, 0, 0, 1]), [], self.sim_step, self.sim_steps_per_env_step, 0, [self.voxel_size / 2, self.voxel_size / 2, self.voxel_size / 2])
            voxel.build()
            self.obstacle_objects.append(voxel)
            self.voxels.append(voxel)

        self.active_objects = self.obstacle_objects  # in this env, there is no appreciable difference between the two
        
    def reset(self, success_rate: float):
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []
        self.ee_starting_points = []

        # pick an experiment at random
        experiment = np.random.choice(self.experiments, p=self.experiments_weights)

        # set up the chosen experiment
        eval('self._set_up_exp' + str(experiment) + '()')
        
        # frame 150: (0.4, 0.0, 0.25), (0.4, 0.0, 0.5), (0.65, 0.15, 0.35)
        # frame 200: (0.3, 0.0, 0.5), (0.5, 0.0, 0.5), (0.4, 0.0, 0.15)
        # frame 280: (0.6, 0.0, 0.15), (0.4, 0.0, 0.5), (0.4, 0.35, 0.25)
        # frame 350: (0.6, 0.0, 0.3), (0.4, 0.0, 0.5), (0.4, 0.35, 0.45)
        # frame 440: (0.6, 0.0, 0.3), (0.35, 0.5, 0.25) 
        # frame 560: (0.35, 0.2, 0.15), (0.35, 0.2, 0.45)

    def _set_up_exp0(self):
        # determine the part of the recording we're going to use for the next episode
        num_frames = round(np.random.uniform(low=self.min_num_frames, high=self.max_num_frames))
        self.start_frame = int(np.random.uniform(low=0, high=len(self.voxel_frames) - num_frames - 1))
        self.current_frame = self.start_frame
        self.end_frame = self.start_frame + num_frames
        self.time = 0

        # set up the scene of the first frame
        self.update()
        # now find some valid goal and start position using helper methods
        val = False
        while not val:
            val = self._create_ee_starting_points(self.robots)
            if not val:
                continue
            val = self._create_position_and_rotation_targets(self.robots)

    def _set_up_exp1(self):
        scenario_list = [
            (150, [np.array([0.4, 0.0, 0.25]), np.array([0.4, 0.0, 0.5]), np.array([0.65, 0.15, 0.35])]),
            (200, [np.array([0.3, 0.0, 0.5]), np.array([0.5, 0.0, 0.5]), np.array([0.4, 0.0, 0.15])]),
            (280, [np.array([0.6, 0.0, 0.15]), np.array([0.4, 0.0, 0.5]), np.array([0.4, 0.35, 0.25])]),
            (350, [np.array([0.6, 0.0, 0.3]), np.array([0.4, 0.0, 0.5]), np.array([0.4, 0.35, 0.45])]),
            (440, [np.array([0.6, 0.0, 0.3]), np.array([0.35, 0.5, 0.25])]),
            (560, [np.array([0.35, 0.2, 0.15]), np.array([0.35, 0.2, 0.45])])
        ]
        scenario = choice(scenario_list)
        self.start_frame = scenario[0]
        self.current_frame = self.start_frame
        self.end_frame = len(self.voxel_frames) - 1
        self.time = 0

        # set up the scene of the first frame
        self.update()
        # set target
        self.position_targets = [choice(scenario[1])]
        # set robot to default position
        self.robots[0].moveto_joints(self.robots[0].resting_pose_angles, False)


    def update(self):
        self.time += self.sim_step * self.sim_steps_per_env_step
        # need to do this, visual performance is terrible otherwise
        pyb_u.toggle_rendering(False)
        # move the voxels
        for idx, voxel_center in enumerate(self.voxel_frames[self.current_frame]):
            # as long as we have enough voxels, move them to respective entry
            if idx < len(self.voxels):
                self.voxels[idx].move_base(voxel_center)
        # in case there were less voxels in this frame, move the rest of the available voxels out of sight
        for i in range(idx, len(self.voxels)):
            self.voxels[i].move_base(self.storage)
        potential_next_frame_time = (self.current_frame + 1 - self.start_frame) * self.frame_time
        if self.time > potential_next_frame_time:
            if self.current_frame == self.end_frame:
                self.current_frame = self.start_frame
                self.voxel_frames = list(reversed(self.voxel_frames))
                self.time = 0
            else:
                self.current_frame += 1
        pyb_u.toggle_rendering(True)