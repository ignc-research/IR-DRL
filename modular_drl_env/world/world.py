from abc import ABC, abstractmethod
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import numpy as np


class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int):

        # list that will contain all  object ids with collision managed by this world simulation
        self.objects_ids = []
        # list that will contain all purely visual object ids like spheres and cubes without collision
        self.aux_object_ids = []
        # same, but for lines
        self.aux_lines = []
        # list that will contain all obstacle objects used in this world
        self.obstacle_objects = []

        # set sim step
        self.sim_step = sim_step

        # set env id, this is a number that differentiates multiple envs running in parallel in training, necessary for file related stuff
        self.env_id = env_id

        # set up workspace boundaries
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = workspace_boundaries

        # targets for goals that need to interact with the world
        # fill these in the build method
        self.position_targets = []
        self.rotation_targets = []
        self.joints_targets = []

        # can be used to create starting points for robots (see method below), but is not accessed by anything else outside of this class
        self.ee_starting_points = []

        # list of robots, gets filled by register method down below
        self.robots = []  # all robots in world

        # flags such that the automated processes elsewhere can recognize what sort of goals this world can support
        # set these yourself if they apply in a subclass
        self.gives_a_position = False
        self.gives_a_rotation = False

    def register_robots(self, robots):
        """
        This method receives a list of robot objects from the outside and sorts the robots therein into a list that is important for other methods.
        Also gives each robot an id int that can be used to identify it and corresponds with its position in the list.
        """
        id_counter = 0
        for robot in robots:
            self.robots.append(robot)
            robot.mgt_id = id_counter
            id_counter += 1

    @abstractmethod
    def build(self, success_rate: float):
        """
        This method should build all the components that make up the world simulation aside from the robot.
        This includes URDF files as well as objects created by code.
        This method should also generate valid targets for all robots that need them, valid starting points and it should also move them there to start the episode.
        Additionally, this method receives the success rate of the gym env as a value between 0 and 1. You could use this to
        set certain parameters, e.g. to the make world become more complex as the agent's success rate increases.

        All object ids loaded in by this method must be added to the self.object_ids list! Otherwise they will be ignored in collision detection.
        If you use our Obstacle objects, add them (the objects themselves, not just their object ids) to self.obstacle_objects.
        """
        pass

    @abstractmethod
    def reset(self, success_rate: float):
        """
        This method should reset all lists, arrays, variables etc. that handle the world to such a state that a new episode can be run.
        Meaning that after this method is done, build() can be called again.
        Don't reset the simulation itself, that will be handled by the gym env.
        Additionally, this method receives the success rate of the gym env as a value between 0 and 1. You could use this to
        set certain parameters, e.g. to the make world become more complex as the agent's success rate increases.
        """
        pass

    def build_visual_aux(self):
        """
        This method should add objects that are not necessary to the purpose of the world and useful only for visual quality.
        By default, this method will mark the workspace boundaries, but you can extend it in your subclasses as you need.
        Visual objects related to a goal should be implemented by that goal.
        (This is because the world should be usable with all sorts of goals, even those that need different visualizations for their goals,
        e.g. a target sphere vs. a target cube)
        Objects built here should NOT be added to self.object_ids but to self.aux_object_ids. Dont forget to reset self.aux_object_ids in your reset methods.
        """
        line_starts = [
            [self.x_min, self.y_min, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_max],
            [self.x_min, self.y_min, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_min, self.y_min, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_min, self.y_min, self.z_min],
            [self.x_max, self.y_min, self.z_min]
        ]
        line_ends = [
            [self.x_min, self.y_min, self.z_max],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_min, self.y_max, self.z_max],
            [self.x_max, self.y_max, self.z_max],
            [self.x_max, self.y_min, self.z_min],
            [self.x_max, self.y_max, self.z_min],
            [self.x_min, self.y_max, self.z_min],
            [self.x_max, self.y_max, self.z_min]
        ]
        colors = [[1, 1, 1] for _ in line_starts]

        self.aux_object_ids += pyb_u.draw_lines(line_starts, line_ends, colors)
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

    def _create_ee_starting_points(self, robots, custom_joints_limits=[], factor=-1, base_dist=7.5e-2) -> None:
        """
        This is a helper method to generate valid starting points for all robots in the env. You can use this within your build method to generate starting positions at random.
        The robots parameter is a list of all robots that should be touched by this method.
        The custom joints_limits parameter is a list of tuples, where entry one is custom lower limits and entry two is custom upper limits.
        If used at all (i.e. its not an empty list) it must be of the same length as robots. If you don't want custom limits for a robot, set one or both of respective tuple entries to None.
        Factor is a float between 0 and 1 that can be used to steer the generation of random starting angles, see below in the code for what it does.
        Base dist is a float that determines a sphere around the robot base in which no positions can be spawned in.
        """
        counter = 0
        val = False
        while not val and counter < 10000:
            joints = []
            counter += 1
            # first generate random pose for each robot and set it
            oob = False
            too_close_to_base = False
            for idx, robot in enumerate(robots):
                joint_dim = robot.get_action_space_dims()[0]
                if custom_joints_limits:
                    lower_limits = custom_joints_limits[idx][0]
                    lower_limits = robot.joints_limits_lower if lower_limits is None else lower_limits
                    upper_limits = custom_joints_limits[idx][1]
                    upper_limits = robot.joints_limits_upper if upper_limits is None else upper_limits
                else:
                    lower_limits = robot.joints_limits_lower
                    upper_limits = robot.joints_limits_upper
                if factor == -1:
                    random_joints = np.random.uniform(low=lower_limits, high=upper_limits, size=(joint_dim,))
                else:
                    random_joints = (1 - factor) * robot.resting_pose_angles + factor * np.random.uniform(low=lower_limits, high=upper_limits, size=(joint_dim,))
                    upper_limit_mask = random_joints > upper_limits
                    lower_limit_mask = random_joints < lower_limits
                    random_joints[upper_limit_mask] = upper_limits[upper_limit_mask]
                    random_joints[lower_limit_mask] = lower_limits[lower_limit_mask]
                joints.append(random_joints)
                robot.moveto_joints(random_joints, False)

                # check if robot is out of bounds directly
                robot.position_rotation_sensor.reset()  # refresh position data for oob calculation below
                if self.out_of_bounds(robot.position_rotation_sensor.position):
                    oob = True
                    break
                if np.linalg.norm(robot.position_rotation_sensor.position - robot.base_position) < base_dist:
                    too_close_to_base = True
                    break
            # if out of bounds, start over
            if oob or too_close_to_base:
                continue
            # now check if there's a collision
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            # if so, start over
            if pyb_u.collision:
                continue
            # if we reached this line, then everything works out
            val = True
        if val:
            counter = 0
            for robot in self.robots:
                if robot in robots:  # robot is to be considered
                    pos = robot.position_rotation_sensor.position
                    rot = robot.position_rotation_sensor.rotation
                    self.ee_starting_points.append((pos, rot, joints[counter]))
                    counter += 1
                else:  # other robots (e.g. camera arm robot)
                    self.ee_starting_points.append((None, None, None))    
        return val

    def _create_position_and_rotation_targets(self, robots, min_dist=0, custom_joints_limits=[], base_dist=7.5e-2) -> None:
        """
        This is a helper method to generate valid targets for your robots with goals at random. You can use this in your build method to generate targets.
        The min_dist parameter will enforce a minimum cartesian distance to the respective robots starting point.
        For the custom joints parameter see above.
        """
        counter = 0
        val = False
        while not val and counter < 10000:
            counter += 1
            # first generate random pose for each robot and set it
            oob_or_too_close = False
            too_close_to_base = False
            for idx, robot in enumerate(robots):
                joint_dim = robot.get_action_space_dims()[0]
                if custom_joints_limits:
                    lower_limits = custom_joints_limits[idx][0]
                    lower_limits = robot.joints_limits_lower if lower_limits is None else lower_limits
                    upper_limits = custom_joints_limits[idx][1]
                    upper_limits = robot.joints_limits_upper if upper_limits is None else upper_limits
                else:
                    lower_limits = robot.joints_limits_lower
                    upper_limits = robot.joints_limits_upper
                random_joints = np.random.uniform(low=lower_limits, high=upper_limits, size=(joint_dim,))
                robot.moveto_joints(random_joints, False)

                # check if robot is out of bounds directly
                robot.position_rotation_sensor.reset()  # refresh position data for oob calculation below
                robot.joints_sensor.reset()  # refresh joints data for calculation further down
                if self.out_of_bounds(robot.position_rotation_sensor.position):
                    oob_or_too_close = True
                    break
                # check if position is too close to starting position
                if np.linalg.norm(robot.position_rotation_sensor.position - self.ee_starting_points[robot.id][0]) < min_dist:
                    oob_or_too_close = True
                    break
                if np.linalg.norm(robot.position_rotation_sensor.position - robot.base_position) < base_dist:
                    too_close_to_base = True
                    break
            # if out of bounds or too close, start over
            if oob_or_too_close or too_close_to_base:
                continue
            # now check if there's a collision
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            # if so, start over
            if pyb_u.collision:
                continue
            # if we reached this line, then everything works out
            val = True
        if val:
            for idx, robot in enumerate(self.robots):
                if robot in robots:  # robot is to be considered
                    pos = robot.position_rotation_sensor.position
                    rot = robot.position_rotation_sensor.rotation
                    joints = robot.joints_sensor.joints_angles
                    self.position_targets.append(pos)
                    self.rotation_targets.append(rot)
                    self.joints_targets.append(joints)
                    # reset robot back to starting position
                    robot.moveto_joints(self.ee_starting_points[idx][2], False)
                else:  # other robots (e.g. camera arm robot)
                    self.position_targets.append(None)
                    self.rotation_targets.append(None)   
                    self.joints_targets.append(None)
        return val

    def out_of_bounds(self, position: np.ndarray) -> bool:
        """
        Helper method that returns whether a given position is within the workspace bounds or not.
        """
        x, y, z = position
        if x > self.x_max or x < self.x_min:
            return True
        elif y > self.y_max or y < self.y_min:
            return True
        elif z > self.z_max or z < self.z_min:
            return True
        return False
