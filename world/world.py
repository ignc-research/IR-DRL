from abc import ABC, abstractmethod

class World(ABC):
    """
    Abstract Base Class for a simulation world. Methods signed with abstractmethod need to be implemented by subclasses.
    See the random obstacles world for examples.
    """

    def __init__(self, workspace_boundaries:list):

        # set initial build state
        self.built = False

        # list that will contain all PyBullet object ids managed by this world simulation
        self.objects_ids = []

        # set up workspace boundaries
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = workspace_boundaries


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
        This method should add objects that are completely not necessary to the purpose of the world and useful only for visual quality.
        This could include things like lines marking the boundaries of the workspace or geometry marking a target zone etc.
        Objects built here should NOT be added to self.object_ids
        """
    
    @abstractmethod
    def update(self):
        """
        This method should update all dynamic and movable parts of the world simulation. If there are none it doesn't need to do anything at all.
        """
        pass

