from abc import ABC, abstractmethod
import numpy as np

class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries:list, robot_base_positions:list, robot_base_orientations:list):

        # set initial build state
        self.built = False

        # list that will contain all PyBullet object ids managed by this world simulation
        self.objects_ids = []

        # set up workspace boundaries
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = workspace_boundaries

        # targets for goals that need to interact with the world
        self.position_targets = None
        self.rotation_targets = None

        # robot base points
        self.robot_base_positions = robot_base_positions
        self.robot_base_orientations = robot_base_orientations

        # points for robot end effectors at episode start
        self.ee_starting_points = None

        # list of robots, gets filled by register method down below
        self.robots_in_world = []  # all robots in world
        self.robots_with_position = []  # all robots with the position goal
        self.robots_with_orientation = []  # all robots with the orientation goal

    def register_robots(self, robots):
        """
        This method receives a list of robot objects from the outside and sorts the robots therein into several lists that are important for
        other methods.
        """
        for robot in robots:
            self.robots_in_world.append(robot)
            for goal in robot.goals:
                if goal.name == "position":
                    self.robots_with_position.append(robot)
                elif goal.name == "orientation":
                    self.robots_with_orientation.append(robot)
    @abstractmethod
    def build(self):
        """
        This method should build all the components that make up the world simulation aside from the robot.
        This includes URDF files as well as objects created by PyBullet code.
        All object ids loaded in by this method must be added to the self.object_ids list! Otherwise they will be ignored in collision detection.
        If the self.built variable is True, this method should do nothing. If the method finishes, it should set self.built to True.
        """
        pass

    @abstractmethod
    def build_visual_aux(self):
        """
        This method should add objects that are not necessary to the purpose of the world and useful only for visual quality.
        This could include things like lines marking the boundaries of the workspace or geometry marking a target zone etc.
        Objects built here should NOT be added to self.object_ids
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

    @abstractmethod
    def _create_ee_starting_points(self) -> list:
        """
        This method should return a valid starting position for the end effector at episode start.
        Valid meaning reachable and not in collision.
        The return should be a list of tuples, each containing a 3D Point and a quaternion (which can be None instead if no specific rotation is needed), one tuple for each robot registered in the world.
        """
        pass

    @abstractmethod
    def _create_position_target(self) -> list:
        """
        This method should return a valid target position within the world simulation for a robot end effector.
        Valid meaning (at least very likely) being reachable for the robot without collision.
        The return value should be a list of 3D points as numpy arrays, one each for every robot registered with the world that needs a position target.
        """
        pass

    @abstractmethod
    def _create_rotation_target(self) -> list:
        """
        This method should return a valid target rotation within the world simulation for a robot end effector a
        Valid meaning (at least very likely) being reachable for the robot without collision.
        The return value should be a list of quaternions as numpy arrays, one for each robot registered with the world that needs a rotation target.
        """
        pass
