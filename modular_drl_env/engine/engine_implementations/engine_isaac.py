from ..engine import Engine
import atexit
from typing import List, TYPE_CHECKING, Union, Tuple
import numpy as np
import numpy.typing as npt

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

# Try importing all Issac modules in a try/except to allow compilation without it
try:
    from omni.isaac.kit import SimulationApp
except ImportError:
    pass


class IsaacEngine(Engine):
    def __init__(self, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: list, assets_path: str) -> None:
        super().__init__("Isaac", use_physics_sim, display_mode, sim_step, gravity, assets_path)
        # setup simulation
        self.simulation = SimulationApp({"headless": not display_mode})

        # terminate simulation once program exits
        atexit.register(self.simulation.close)

    ###################
    # general methods #
    ###################

    def step(self):
        # simulate physics step if pyhsics is enabled
        if self.use_physics_sim:
            self.simulation.update()
            
    def reset(self):
        """
        This method should reset the entire simulation, meaning that all objects should be deleted and everything be reset.
        """
        raise "Not implemented!"

    def perform_collision_check(self, robots: List["Robot"], obstacles: List[int]) -> bool:
        """
        Performs a collision check 
        1. between all robots and all obstacles in the world and
        2. between each robot
        """
        raise "Not implemented!"

    ####################
    # geometry methods #
    ####################

    # all orientations are unit quaternions in x, y, z, w format
    # all colors are RGB values between 0 and 1 in r, g, b, a format

    def add_ground_plane(self, position: np.ndarray) -> int:
        """
        Spawns a ground plane into the world at position. 
        Must return a unique int identifying the ground plane within the engine.
        """
        raise "Not implemented!"

    def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: float=1) -> int:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique int identifying the newly spawned object within the engine.
        """
        raise "Not implemented!"

    def create_box(self, position: np.ndarray, orientation: np.ndarray, mass: float, halfExtents: List, color: List[float], collision: bool=True) -> int:
        """
        Spawns a box at position and orientation. Half extents are the length of the three dimensions starting from position.
        Must return a unique int identifying the newly spawned object within the engine.
        """
        raise "Not implemented!"

    def create_sphere(self, position: np.ndarray, radius: float, mass: float, color: List[float], collision: bool=True) -> int:
        """
        Spawns a sphere.
        Must return a unique int identifying the newly spawned object within the engine.
        """
        raise "Not implemented!"

    def create_cylinder(self, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, color: List[float], collision: bool=True) -> int:
        """
        Spawns a cylinder.
        Must return a unique int identifying the newly spawned object within the engine.
        """
        raise "Not implemented!"

    def move_base(self, object_id: int, position: np.ndarray, orientation: np.ndarray):
        """
        Moves the base of an object or robot towards the desired position and orientation instantaneously, without physcis calucations.
        """
        raise "Not implemented!"

    ######################################################
    # helper methods (e.g. lines or debug visualization) #
    ######################################################

    def add_aux_line(self, lineFromXYZ: List[float], lineToXYZ: List[float]) -> int:
        """
        Adds a simple line
        """
        raise "Not implemented!"

    def remove_aux_object(self, aux_object_id):
        """
        Removes an auxillary object via its int id.
        """
        raise "Not implemented!"

    #################
    # robot methods #
    #################

    def joints_torque_control_velocities(self, robot_id: int, joints_ids: List[int], target_velocities: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the velocities of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        raise "Not implemented!"

    def joints_torque_control_angles(self, robot_id: int, joints_ids: List[int], target_angles: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the angles of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        raise "Not implemented!"

    def set_joint_value(self, robot_id: int, joint_id: int, joint_value: float):
        """
        Sets the a specific joint to a specific value ignoring phycis, i.e. resulting in instant movement.
        """
        raise "Not implemented!"

    def get_joint_value(self, robot_id: int, joint_id: int) -> float:
        """
        Returns the value a single joint is at.
        """
        raise "Not implemented!"
    
    def get_link_state(self, robot_id: int, link_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple with position and orientation, both in world frame, of the link in question.
        """
        raise "Not implemented!"

    def solve_inverse_kinematics(self, robot_id: int, end_effector_link_id: int, target_position: np.ndarray, target_orientation: Union[np.ndarray, None], max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
        """
        Solves the inverse kinematics problem for the given robot. Returns a vector of joint values.
        If target_orientation is None perform inverse kinematics for position only.
        """
        raise "Not implemented!"

    def get_joints_ids_actuators(self, robot_id) -> List[int]:
        """
        This should return a List uniquely identifying (per robot) ints for every joint that is an actuator, e.g. revolute joints but not fixed joints.
        """
        raise "Not implemented!"