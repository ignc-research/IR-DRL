from ..engine import Engine
import atexit
from typing import List, TYPE_CHECKING, Union, Tuple
import numpy as np
import numpy.typing as npt
from modular_drl_env.isaac_bridge.bridge import is_isaac_running

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

# Try importing all Issac modules in a try/except to allow compilation without it
try:
    from omni.isaac.kit import SimulationApp
    from omni.isaac.urdf import _urdf
    from omni.kit.commands import execute
except ImportError:
    # raise error only if Isaac is running
    if is_isaac_running():
        raise
    # Isaac is not enabled, ignore exception
    pass


class IsaacEngine(Engine):
    def __init__(self, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: list, assets_path: str) -> None:
        super().__init__("Isaac", use_physics_sim, display_mode, sim_step, gravity, assets_path)
        # setup simulation
        self.simulation = SimulationApp({"headless": not display_mode})

        # terminate simulation once program exits
        atexit.register(self.simulation.close)

        # save asset path
        self.assets_path = assets_path

        # configure URDF importer # todo: why is this necessary?
        urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        import_config.default_drive_strength = 1047.19751
        import_config.default_position_drive_damping = 52.35988
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 1
        import_config.density = 0.0

        self.import_config = import_config

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
        # todo: track objects when adding them, remove them here
        pass

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
        # get absolute path to urdf file
        path = self.get_absolute_asset_path(urdf_path)

        # import urdf
        result, prim_path = execute("URDFParseAndImportFile", urdf_path=path, import_config=self.import_config,)

        print(result, prim_path)

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
    
    def get_absolute_asset_path(self, path:str) -> str:
        return self.assets_path + path