from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Union, Tuple
import numpy as np
import numpy.typing as npt

# Use type checking to enable type hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.robot.robot import Robot
    from modular_drl_env.world.obstacles.obstacle import Obstacle

# variable that will hold the specific engine when it's initialized later on
# made available to other files via get_instance (see below)
_instance = None

class Engine(ABC):
    """
    Abstract base class that handles calls to physics engine methods in the main environment.py file.
    This does not include specific e.g. sensor or robot implementations, these are handled by their own subclasses.
    """

    def __init__(self, engine_type: str, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: List, assets_path: str) -> None:
        """
        This method starts the engine in the python code.
        It should also set several attributes using the parameters:
        - use_physics_sim: bool that determines the way objecst move in the simulation: instantaneous (False) or according to the physcis simulation (True)
        - display_mode: a bool that determines whether to render a GUI for the user (True) or not (False)
        - sim_step: a float number that determines the sim time that passes with call of Engine.step
        - gravity: a 3-vector that determines the gravitational force along the world xyz-axes
        - assets_path: a string containing the absolute path of the assets folder from where the engine will load in meshes
        """
        super().__init__()
        self.engine_type = engine_type  # we use this to check for cases when a feature is not supported by some engine types
        self.use_physics_sim = use_physics_sim  # determines how objects are moved within the engine, either by brute setting their position or correct physics simulation

    ###################
    # general methods #
    ###################

    @abstractmethod
    def step(self):
        """
        This method should simply let sim time pass within the engine. This should not apply commands or forces to objects in the simulation on its own.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method should reset the entire simulation, meaning that all objects should be deleted and everything be reset.
        """
        pass

    @abstractmethod
    def perform_collision_check(self, robots: List["Robot"], obstacles: List[str]) -> bool:
        """
        Performs a collision check 
        1. between all robots and all obstacles in the world and
        2. between each robot
        """
        pass

    ####################
    # geometry methods #
    ####################

    # all orientations are unit quaternions in x, y, z, w format
    # all colors are RGB values between 0 and 1 in r, g, b, a format

    @abstractmethod
    def add_ground_plane(self, position: np.ndarray) -> str:
        """
        Spawns a ground plane into the world at position. 
        Must return a unique str identifying the ground plane within the engine.
        """
        pass

    @abstractmethod
    def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: float=1, is_robot: bool=False) -> str:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique str identifying the newly spawned object within the engine.
        The is_robot flag determines whether the engine handles this object as a robot (something with movable links/joints) or a simple geometry object (a singular mesh).
        """
        pass

    @abstractmethod
    def create_box(self, position: np.ndarray, orientation: np.ndarray, mass: float, halfExtents: List, color: List[float], collision: bool=True) -> str:
        """
        Spawns a box at position and orientation. Half extents are the length of the three dimensions starting from position.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        pass

    @abstractmethod
    def create_sphere(self, position: np.ndarray, radius: float, mass: float, color: List[float], collision: bool=True) -> str:
        """
        Spawns a sphere.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        pass

    @abstractmethod
    def create_cylinder(self, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, color: List[float], collision: bool=True) -> str:
        """
        Spawns a cylinder.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        pass

    @abstractmethod
    def move_base(self, object_id: str, position: np.ndarray, orientation: np.ndarray):
        """
        Moves the base of an object or robot towards the desired position and orientation instantaneously, without physcis calucations.
        """
        pass

    ######################################################
    # helper methods (e.g. lines or debug visualization) #
    ######################################################

    @abstractmethod
    def add_aux_line(self, lineFromXYZ: List[float], lineToXYZ: List[float], color: List[float]=None) -> str:
        """
        Adds a simple line
        Must return a unique str identifying the newly spawned object within the engine.
        """
        pass

    @abstractmethod
    def remove_aux_object(self, aux_object_id: str):
        """
        Removes an auxillary object via its int id.
        """
        pass

    #################
    # robot methods #
    #################

    @abstractmethod
    def joints_torque_control_velocities(self, robot_id: str, joints_ids: List[int], target_velocities: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the velocities of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        pass

    @abstractmethod
    def joints_torque_control_angles(self, robot_id: str, joints_ids: List[int], target_angles: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the angles of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        pass

    @abstractmethod
    def set_joint_value(self, robot_id: str, joint_id: int, joint_value: float):
        """
        Sets the a specific joint to a specific value ignoring phycis, i.e. resulting in instant movement.
        """
        pass

    def set_joints_values(self, robot_id: str, joints_ids: List[int], joints_values: npt.NDArray[np.float32]):
        """
        Same as set_joint_value, but for multiple joints at once.
        """
        for idx, joint_id in enumerate(joints_ids):
            self.set_joint_value(robot_id, joint_id, joints_values[idx])

    @abstractmethod
    def get_joint_value(self, robot_id: str, joint_id: int) -> float:
        """
        Returns the value a single joint is at.
        """
        pass

    def get_joints_values(self, robot_id: str, joints_ids: List[int]) -> npt.NDArray[np.float32]:
        """
        Same as get_joint_values, but for multiple joints at once.
        """
        values = []
        for joint_id in joints_ids:
            values.append(self.get_joint_value(robot_id, joint_id))
        return np.array(values)
    
    @abstractmethod
    def get_link_state(self, robot_id: str, link_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple with position and orientation, both in world frame, of the link in question.
        """
        pass

    @abstractmethod
    def solve_inverse_kinematics(self, robot_id: str, end_effector_link_id: str, target_position: np.ndarray, target_orientation: Union[np.ndarray, None], max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
        """
        Solves the inverse kinematics problem for the given robot. Returns a vector of joint values.
        If target_orientation is None perform inverse kinematics for position only.
        max_iterations and threshold govern the maximum number of IK iterations and the target precision threshold respectively.
        """
        pass

    @abstractmethod
    def get_joints_ids_actuators(self, robot_id: str) -> List[str]:
        """
        This should return a List of uniquely identifying (per robot) strs for every joint that is an actuator, e.g. revolute joints but not fixed joints.
        """
        pass

    @abstractmethod
    def get_links_ids(self, robot_id: str) -> List[str]:
        """
        This should return a List of uniquely identifying (per robot) strs for every link that makes up the robot.
        """
        pass

    # sensors
    


# import subclasses, has to be done here because only now Engine is defined
from modular_drl_env.engine.engine_implementations import PybulletEngine, IsaacEngine

class_dict = {
            "Pybullet": PybulletEngine,
            "Isaac": IsaacEngine
        }

_instance = None

def initialize_engine(engine_type: str, engine_config: dict):
    global _instance
    _instance = class_dict[engine_type](**engine_config)

# getter method for use from the outside
def get_instance() -> Engine:
    return _instance



