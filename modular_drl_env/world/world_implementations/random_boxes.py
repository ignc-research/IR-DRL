import numpy as np
from numpy.random import shuffle

from modular_drl_env.world.world import World
from modular_drl_env.world.obstacles.shapes import Box
import pybullet_data as pyb_d

__all__ = [
    'RandomBoxesWorld'
]


class RandomBoxesWorld(World):

    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int,
                 box_measurements=None, max_num_boxes=15, obstacle_min_dist=0.2):
        super().__init__(workspace_boundaries, sim_step, env_id)
        if box_measurements is None:
            box_measurements = [0.05, 0.1, 0.05, 0.1, 0.05, 0.1]
        self.obstacle_objects = []  # list to access the obstacle python objects

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, \
            self.box_h_min, self.box_h_max = box_measurements
        self.previous_success_rate = 0
        self.max_num_obstacles = max_num_boxes
        self.obstacle_min_dist = obstacle_min_dist

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for object in self.obstacle_objects:
            del object
        self.obstacle_objects = []
        self.aux_object_ids = []
        self.previous_success_rate = (success_rate + self.previous_success_rate)/2

    def update(self):

        for obstacle in self.obstacle_objects:
            obstacle.move()

    def build(self, success_rate: float):
        # table
        self.objects_ids.append(self.engine.load_urdf(pyb_d.getDataPath() + "/table/table.urdf", np.array([0, 0, -1.1]), np.array([0, 0, 0, 1]), [1.75, 1.75, 1.75]))

        # add ground plate
        self.objects_ids.append(self.engine.add_ground_plane(np.array([0, 0, -1.11])))

        rand_number = int(np.round(self.previous_success_rate * self.max_num_obstacles))

        # add the obstacles
        for i in range(rand_number):
            # generate a random position in the workspace
            #position = np.random.uniform(low=np.array([self.x_min, self.y_min, self.z_min]),
            #                             high=np.array([self.x_max, self.y_max, self.z_max]), size=(3,))

            min_dist = self.obstacle_min_dist
            robot_start = self.robots_in_world[0].base_position
            x = np.random.choice(np.array([np.random.uniform(self.x_min, robot_start[0] - min_dist), np.random.uniform(robot_start[0] + min_dist, self.x_max)]))
            y = np.random.choice(np.array([np.random.uniform(self.y_min, robot_start[0] - min_dist), np.random.uniform(robot_start[0] + min_dist, self.y_max)]))
            z = np.random.uniform(self.z_min, self.z_max)
            move_step = 0
            trajectory = []
            position = np.array([x, y, z])

            # plate
            # generate random size
            length = np.random.uniform(low=self.box_l_min, high=self.box_l_max)
            width = np.random.uniform(low=self.box_w_min, high=self.box_w_max)
            height = np.random.uniform(low=self.box_h_min, high=self.box_h_max)

            # randomly assign lwh to xyz
            dims = [length, width, height]
            shuffle(dims)
            plate = Box(position, [0, 0, 0, 1], trajectory, move_step, dims)
            self.obstacle_objects.append(plate)
            self.objects_ids.append(plate.build())

        # generate starting points and targets
        robots_with_starting_points = [robot for robot in self.robots_in_world if robot.goal is not None]
        val = self._create_ee_starting_points(robots_with_starting_points, factor=success_rate**3)
        if not val:
            return
        min_dist = min((self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2, (self.z_max - self.z_min) / 2)
        val = self._create_position_and_rotation_targets(robots_with_starting_points, min_dist=min_dist)
        if not val:
            return
        
        # with some random chance, generate objects between goal and start
        if success_rate > 0.35:
            pos_robot = self.ee_starting_points[0][0]
            pos_goal = self.position_targets[0]
            collision = True
            self.robots_in_world[0].moveto_joints(self.joints_targets[0], False)
            # generate random obstacle somewhere where it doesnt bother the rest of the scene
            length = np.random.uniform(low=self.box_l_min, high=self.box_l_max) * 0.5
            width = np.random.uniform(low=self.box_w_min, high=self.box_w_max) * 0.5
            height = np.random.uniform(low=self.box_h_min, high=self.box_h_max) * 0.5
            dims = [length, width, height]
            shuffle(dims)
            random_quat = np.random.normal(loc=0, scale=1.0, size=(4,))
            obstacle_plate = Box(np.array([0, 0, -50]), random_quat, [], 0, dims)
            self.obstacle_objects.append(obstacle_plate)
            self.objects_ids.append(obstacle_plate.build())
            # generate random positions in the path between ee and goal
            while collision:
                dist_obstacle = pos_goal + (pos_robot-pos_goal) * np.random.uniform(0.25, 0.75)
                # generate base
                a = (pos_robot-pos_goal) / np.linalg.norm((pos_robot-pos_goal))
                temp_vec = np.random.uniform(low=-1, high=1, size=(3,))
                temp_vec = temp_vec / np.linalg.norm(temp_vec)
                b = np.cross(a, temp_vec)
                b = b / np.linalg.norm(b)
                c = np.cross(a,b)
                c = c / np.linalg.norm(b)
                # set obstacle_pos as linear combi of base without normal_vec
                obstacle_pos = dist_obstacle + b * np.random.uniform(0, 0.15) + c * np.random.uniform(0, 0.15)
                # move obstacle between start and goal pos
                self.engine.move_base(obstacle_plate.object_id, obstacle_pos, random_quat)

                # check collision
                collision = self.perform_collision_check()
                if collision:
                    continue
                else:
                    self.robots_in_world[0].moveto_joints(self.ee_starting_points[0][2], False)
                    collision = self.perform_collision_check()


            #TODO: move robot to goal, spawn plate, then check if there is collision, if yes: spawn new plate, if no append plate
            #self.check_for_collision()

           
        # move robots to starting position
        for idx, robot in enumerate(self.robots_in_world):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)
   
    

    # def _create_position_and_rotation_targets(self, robots, min_dist=0, custom_joints_limits=[]) -> None:
    #     try:
    #         super()._create_position_and_rotation_targets(robots, min_dist=0, custom_joints_limits=[])
    #     except Exception:
    #         self.reset(0.0)
    #         self.build(0.0)
    
    # def _create_ee_starting_points(self, robots, custom_joints_limits=..., factor=-1, base_dist=0.075) -> None:
    #     try:
    #         return super()._create_ee_starting_points(robots, custom_joints_limits, factor, base_dist)
    #     except Exception:
    #         self.reset(0.0)
    #         self.build(0.0)

    # def _create_ee_starting_points(self, robots, custom_joints_limits=[]) -> None:
    #     counter = 0
    #     val = False
    #     while not val and counter < 10000:
    #         joints = []
    #         counter += 1
    #         oob = False
    #         for idx, robot in enumerate(robots):
    #             random_joints = robot.resting_pose_angles
    #             joints.append(random_joints)
    #             robot.moveto_joints(random_joints, False)
    #             robot.position_rotation_sensor.reset()
    #             if self.out_of_bounds(robot.position_rotation_sensor.position):
    #                 oob = True
    #                 break
    #         # if out of bounds, start over
    #         if oob:
    #             continue
    #         # now check if there's a collision
    #         self.perform_collision_check()
    #         # if so, start over
    #         if self.collision:
    #             continue
    #         # if we reached this line, then everything works out
    #         val = True
    #     if val:
    #         counter = 0
    #         for robot in self.robots_in_world:
    #             if robot in robots:  # robot is to be considered
    #                 pos = robot.position_rotation_sensor.position
    #                 rot = robot.position_rotation_sensor.rotation
    #                 self.ee_starting_points.append((pos, rot, joints[counter]))
    #                 counter += 1
    #             else:  # other robots (e.g. camera arm robot)
    #                 self.ee_starting_points.append((None, None, None))
    #         return
