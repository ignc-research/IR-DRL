from abc import ABC, abstractmethod
import numpy as np
import pybullet as pyb


class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int):

        # list that will contain all PyBullet object ids with collision managed by this world simulation
        self.objects_ids = []
        # list that will contain all purely visual PyBullet object ids (e.g. explicatory lines, workspace boundaries etc.)
        self.aux_object_ids = []
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
        self.robots_in_world = []  # all robots in world

        # collision attribute, for convenient outside access
        self.collision = False

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
            self.robots_in_world.append(robot)
            robot.id = id_counter
            id_counter += 1

    def perform_collision_check(self):
        """
        Performs a collision check 
        1. between all robots and all obstacles in the world and
        2. between each robot
        
        Stores the result in a class variable.
        """
        pyb.performCollisionDetection()
        col = False
        # check for each robot with every obstacle
        for robot in self.robots_in_world:
            for obj in self.objects_ids:
                if len(pyb.getContactPoints(robot.object_id, obj)) > 0:
                    col = True 
                    break
            if col:
                break  # this is to immediately break out of the outer loop too once a collision has been found
        # check for each robot with every other one
        if not col:  # skip if another collision was already detected
            for idx, robot in enumerate(self.robots_in_world[:-1]):
                for other_robot in self.robots_in_world[idx+1:]:
                    if len(pyb.getContactPoints(robot.object_id, other_robot.object_id)) > 0:
                        col = True
                        break
                if col:
                    break  # same as above
        self.collision = col

    @abstractmethod
    def build(self):
        """
        This method should build all the components that make up the world simulation aside from the robot.
        This includes URDF files as well as objects created by PyBullet code.
        This method should also generate valid targets for all robots that need them, valid starting points and it should also move them there to start the episode.

        All object ids loaded in by this method must be added to the self.object_ids list! Otherwise they will be ignored in collision detection.
        If you use our Obstacle objects, add them (the objects themselves, not just their pybullet ids) to self.obstacle_objects. You will want to have in them in a list anyway and doing it via this variable ensures compatability with our pybullet recorder.
        """
        pass

    @abstractmethod
    def reset(self, success_rate):
        """
        This method should reset all lists, arrays, variables etc. that handle the world to such a state that a new episode can be run.
        Meaning that after this method is done, build() can be called again.
        Don't reset the PyBullet simulation itself, that will be handled by the gym env.
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
        a = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        b = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        c = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        d = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        e = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        f = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        g = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        h = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        i = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        j = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        k = pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        l = pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])

        self.aux_object_ids += [a, b, c, d, e, f, g, h, i, j, k , l]
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

    def _create_ee_starting_points(self, robots, custom_joints_limits=[]) -> None:
        """
        This is a helper method to generate valid starting points for all robots in the env. You can use this within your build method to generate starting positions at random.
        The robots parameter is a list of all robots that should be touched by this method.
        The custom joints_limits parameter is a list of tuples, where entry one is custom lower limits and entry two is custom upper limits.
        If used at all (i.e. its not an empty list) it must be of the same length as robots. If you don't want custom limits for a robot, set one or both of respective tuple entries to None.
        """
        counter = 0
        val = False
        while not val and counter < 10000:
            joints = []
            counter += 1
            # first generate random pose for each robot and set it
            oob = False
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
                joints.append(random_joints)
                robot.moveto_joints(random_joints, False)
                # check if robot is out of bounds directly
                robot.position_rotation_sensor.reset()  # refresh position data for oob calculation below
                if self.out_of_bounds(robot.position_rotation_sensor.position):
                    oob = True
                    break
            # if out of bounds, start over
            if oob:
                continue
            # now check if there's a collision
            self.perform_collision_check()
            # if so, start over
            if self.collision:
                continue
            # if we reached this line, then everything works out
            val = True
        if val:
            counter = 0
            for robot in self.robots_in_world:
                if robot in robots:  # robot is to be considered
                    pos = robot.position_rotation_sensor.position
                    rot = robot.position_rotation_sensor.rotation
                    self.ee_starting_points.append((pos, rot, joints[counter]))
                    counter += 1
                else:  # other robots (e.g. camera arm robot)
                    self.ee_starting_points.append((None, None, None))    
            return
        else:  # counter too high
            raise Exception("Tried 10000 times to create valid starting positions for the robot(s) without success, maybe check your obstacle generation code.") 

    def _create_position_and_rotation_targets(self, robots, min_dist=0, custom_joints_limits=[]) -> None:
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
            # if out of bounds or too close, start over
            if oob_or_too_close:
                continue
            # now check if there's a collision
            self.perform_collision_check()
            # if so, start over
            if self.collision:
                continue
            # if we reached this line, then everything works out
            val = True
        if val:
            for idx, robot in enumerate(self.robots_in_world):
                if robot in robots:  # robot is to be considered
                    pos = robot.position_rotation_sensor.position
                    rot = robot.position_rotation_sensor.rotation
                    joints = robot.joints_sensor.joints_angles
                    if robot.goal.needs_a_position:
                        self.position_targets.append(pos)
                    else:
                        self.position_targets.append(None)
                    if robot.goal.needs_a_rotation:
                        self.rotation_targets.append(rot)
                    else:
                        self.rotation_targets.append(None)
                    if robot.goal.needs_a_joints_position:
                        self.joints_targets.append(joints)
                    else:
                        self.joints_targets.append(None)
                    # reset robot back to starting position
                    robot.moveto_joints(self.ee_starting_points[idx][2], False)
                else:  # other robots (e.g. camera arm robot)
                    self.position_targets.append(None)
                    self.rotation_targets.append(None)   
                    self.joints_targets.append(None)
            return
        else:  # counter too high
            raise Exception("Tried 10000 times to create valid targets for the robot(s) without success, maybe check your obstacle generation code.") 


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
