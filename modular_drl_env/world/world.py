from abc import ABC, abstractmethod
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet as pyb
import numpy as np
from modular_drl_env.world.obstacles.shapes import *
from modular_drl_env.world.obstacles.urdf_object import URDFObject


class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries: list, sim_step: float, env_id: int, assets_path: str):

        # list that will contain all purely visual object ids like spheres and cubes without collision
        self.aux_objects = []
        # same, but for lines
        self.aux_lines = []
        # list that will contain all obstacle objects used in this world, even those that are not being used for the current episode
        self.obstacle_objects = []
        # this list should contain those objects that are used in the current episode, this is important to manage as
        # a) our logging will only take these objects into account and b) some collision checks like the RRT planner
        # will only check for objects in this list
        self.active_objects = []

        # set sim step
        self.sim_step = sim_step

        # str that contains the asset path in case the world needs access to those resources
        self.assets_path = assets_path

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
    def set_up(self):
        """
        This method should:
        - build all the components that make up the 3D scene aside from the robots, including meshes, geometric shapes like boxes or other things
            - it should ideally do this in a way that no other geometry needs to be spawned ever again, e.g. if you want a scenario where randomized geometry is spawned on every episode start
              generate enough random variations in this method and then use the reset method (see below) to move a share of the pre-generated geometry to its position on episode start
              while leaving the rest of the pre-generated stuff in some inaccessible location
              (this will save massive amounts of performance, however if necessary you can also delete old and spawn new geometry in reset)
              (look at random_obstacles for an example of how to do it)
            - further, you MUST use the obstacle class for spawning stuff and put all obstacles into the self.obstacle_objects list
        - generate valid goal targets for all robots that need them, e.g. xyz target coordinates, target rotations or target joint configurations
        """
        pass

    @abstractmethod
    def reset(self, success_rate: float):
        """
        This method should:
        - set up the scene such that a new episode can start
            - if some change to the geometry of the scene is necessary, then ideally (see set_up method comments) any such change should rely on pre-generated geometry,
              e.g. by existing moving obstacles to new places or using pre-generated ones that were put in some storage location
            - if absolutely necessary, it's okay to spawn new or delete old geometry, but this should be kept to a minimum to avoid performance impacts
            - ideally, whatever set up, movement and spawning you do perform should lead to a valid starting position of everything, i.e. meaning that there are no immediate collisions and all robots' goals are achievable 
        - move robots into their starting positions
        - generate valid targets as needed by the robots' goals
        - use, if wanted, the success rate in some way
            - this is a number between 0 and 1 that expresses the average rate of success of a number of past episodes, you could use it to progressively make the scenario harder for the robot for example
        """
        pass

    def build_visual_aux(self):
        """
        This method should:
        - add objects that are not necessary for the purpose of scenario and useful only visualizing some aspect of it for the user
        - add any such objects to the self.aux_lines if it's lines or self.aux_objects if it's anything else (pybullet handles these differently)
        By default, this method will draw the workspace boundaries, you can overwrite with additional or entirely new functionality in your subclass.
        Note: this method will only be called once, so stuff done here will last for the entire runtime.
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

        self.aux_lines += pyb_u.draw_lines(line_starts, line_ends, colors)
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

    def get_data_for_logging(self):
        """
        This method logs the position and sizes of all active geometry. Additionally, it will report the closest distance of all robots to the obstacles.
        """
        log_dict = dict()
        for robot in self.robots:
            log_dict[robot.name + "_closestObstDistance_robot"] = np.inf
            log_dict[robot.name + "_closestObstDistance_ee"] = np.inf
        obstacle_log_list = []
        for obstacle in self.active_objects:
            # check distance of obstacle to all robots
            for robot in self.robots:
                obstacle_pyb_id = pyb_u.to_pb(obstacle.object_id)
                robot_pyb_id = pyb_u.to_pb(robot.object_id)
                closestDistances_robot = pyb.getClosestPoints(robot_pyb_id, obstacle_pyb_id, 99)
                closestDistance_robot = min([value[8] for value in closestDistances_robot])
                closestDistance_ee = pyb.getClosestPoints(robot_pyb_id, obstacle_pyb_id, 99, pyb_u.pybullet_link_ids[robot.object_id, robot.end_effector_link_id])[0][8]
                log_dict[robot.name + "_closestObstDistance_robot"] = min(closestDistance_robot, log_dict[robot.name + "_closestObstDistance_robot"])
                log_dict[robot.name + "_closestObstDistance_ee"] = min(closestDistance_ee, log_dict[robot.name + "_closestObstDistance_ee"])

            if type(obstacle) == Sphere:
                obstacle_log_list.append(["Sphere", obstacle.radius, obstacle.position])
            elif type(obstacle) == Box:
                obstacle_log_list.append(["Box", obstacle.halfExtents, obstacle.position, obstacle.orientation])
            elif type(obstacle) == Cylinder:
                obstacle_log_list.append(["Cylinder", obstacle.radius, obstacle.height, obstacle.position, obstacle.orientation])
            elif type(obstacle) == URDFObject:
                obstacle_log_list.append(["URDF", obstacle.urdf_path, obstacle.position, obstacle.orientation])
        log_dict["obstacles"] = obstacle_log_list

        return log_dict

    def _create_ee_starting_points(self, robots, factor=-1, base_dist=7.5e-2) -> None:
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
                joint_dim = len(robot.all_joints_ids)
                if factor == -1:
                    random_joints = robot.sample_valid_configuration()
                else:
                    random_joints = (1 - factor) * robot.resting_pose_angles + factor * robot.sample_valid_configuration()
                    upper_limit_mask = random_joints > robot.joints_limits_upper
                    lower_limit_mask = random_joints < robot.joints_limits_lower
                    random_joints[upper_limit_mask] = robot.joints_limits_upper[upper_limit_mask]
                    random_joints[lower_limit_mask] = robot.joints_limits_lower[lower_limit_mask]
                joints.append(random_joints)
                robot.moveto_joints(random_joints, False, robot.controlled_joints_ids)

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

    def _create_position_and_rotation_targets(self, robots, min_dist=0, base_dist=7.5e-2) -> None:
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
                random_joints = robot.sample_valid_configuration()
                if True:  # replace by some condition that checks if a robot has an aliased joint range
                    random_joints = (random_joints + np.pi) % (2 * np.pi) - np.pi
                robot.moveto_joints(random_joints, False)

                # check if robot is out of bounds directly
                robot.position_rotation_sensor.reset()  # refresh position data for oob calculation below
                robot.joints_sensor.reset()  # refresh joints data for calculation further down
                if self.out_of_bounds(robot.position_rotation_sensor.position):
                    oob_or_too_close = True
                    break
                # check if position is too close to starting position
                if np.linalg.norm(robot.position_rotation_sensor.position - self.ee_starting_points[robot.mgt_id][0]) < min_dist:
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
                    robot.moveto_joints(self.ee_starting_points[idx][2], False, robot.controlled_joints_ids)
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
