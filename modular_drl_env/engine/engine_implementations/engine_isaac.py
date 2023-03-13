from ..engine import Engine
import atexit
from typing import List, TYPE_CHECKING, Union, Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path
from modular_drl_env.isaac_bridge.bridge import is_isaac_running

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

# Try importing all Issac modules in a try/except to allow compilation without it
try:
    # isaac imports may only be used after SimulationApp is started (ISAAC uses runtime plugin system)
    from omni.isaac.kit import SimulationApp
    simulation = SimulationApp({"headless": True})
    # terminate simulation once program exits
    atexit.register(simulation.close)

    from omni.kit.commands import execute
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCylinder
    from omni.usd._usd import UsdContext
    from pxr.Usd import Prim

    class IsaacEngine(Engine):
        def __init__(self, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: list, assets_path: str) -> None:
            super().__init__("Isaac", use_physics_sim, display_mode, sim_step, gravity, assets_path)
            # todo: start simulation here & input display_mode
            # todo: configure gravity

            # make sure the simulation was started
            assert simulation != None, "Issac Sim failed to start!"

            # retrieve interfaces allowing to access ISAAC
            self.simulation = simulation
            self.context: UsdContext = simulation.context
            self.app = simulation.app

            # create a world, allowing to spawn objects
            self.world = World(physics_dt=sim_step)
            self.scene = self.world.scene
            self.stage = self.world.stage

            assert self.world != None, "Isaac world failed to load!"
            assert self.scene != None, "Isaac scene failed to load!"
            assert self.stage != None, "Isaac stage failed to load!"

            # save asset path
            self.assets_path = assets_path

            # configure URDF importer
            result, self._config = execute("URDFCreateImportConfig")
            if not result:
                raise "Failed to create URDF import config"

            # Set defaults in import config
            self._config.merge_fixed_joints = False
            self._config.convex_decomp = False
            self._config.import_inertia_tensor = True
            self._config.fix_base = False

            # Tracks which id corresponds with which spawned modifyable object
            self._articulations: dict[str, Articulation] = {}
            # Tracks spawned cubes
            self._cubes: dict[str, DynamicCuboid] = {}
            # Tracks spawned spheres
            self._spheres: dict[str, DynamicSphere] = {}
            # Track spawned cylinders
            self._cylinders: dict[str, DynamicCylinder] = {}
            

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
            # todo: this calles reset function on objects. Enough?
            self.world.reset()

        def perform_collision_check(self, robots: List["Robot"], obstacles: List[str]) -> bool:
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

        def add_ground_plane(self, position: np.ndarray) -> str:
            """
            Spawns a ground plane into the world at position. 
            Must return a unique str identifying the ground plane within the engine.
            """

            # define prim path
            name = "defaultGroundPlane"
            prim_path = "/World/" + name

            # add object to world
            self.world.scene.add_default_ground_plane(prim_path=prim_path, z_position=position[2], name=name)

            # return name as id
            return name
        

        def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: float=1) -> str:
            """
            Loads in a URDF file into the world at position and orientation.
            Must return a unique str identifying the newly spawned object within the engine.
            """
            # get absolute path to urdf
            urdf_path = self.get_absolute_asset_path(urdf_path)

            # import URDF
            from omni.kit.commands import execute
            success, prim_path = execute("URDFParseAndImportFile", urdf_path=urdf_path, import_config=self._config, )
            
            # make sure import succeeded
            assert success, "Failed urdf import of: " + urdf_path

            # create articulation view and give object an id
            obj, id = self.add_urdf_to_scene(prim_path)
            
            # set position, orientation, scale of loaded obj
            obj.set_world_pose(position, orientation)
            obj.set_local_scale([scale, scale, scale])

            return id
            

        def create_box(self, position: np.ndarray, orientation: np.ndarray, mass: float, halfExtents: List, color: List[float], collision: bool=True) -> str:
            """
            Spawns a box at position and orientation. Half extents are the length of the three dimensions starting from position.
            Must return a unique str identifying the newly spawned object within the engine.
            """

            # generate unique prim path
            name = "cube" + str(len(self._cubes))
            prim_path = "/World/" + name

            # create cube # todo: what is halfExtens?
            obj = DynamicCuboid(prim_path, position=position, orientation=orientation, mass=mass, color=self.to_isaac_color(color), name=name)
            obj.set_collision_enabled(collision)

            # add cube to scene
            self.scene.add(obj)

            # track object
            self._cubes[name] = obj
            return name

        def create_sphere(self, position: np.ndarray, radius: float, mass: float, color: List[float], collision: bool=True) -> str:
            """
            Spawns a sphere.
            Must return a unique str identifying the newly spawned object within the engine.
            """
            name = "sphere" + str(len(self._spheres))
            prim_path = "/World/" + name

            # create sphere
            obj = DynamicSphere(prim_path, position=position, mass=mass, color=self.to_isaac_color(color), radius=radius, name=name)
            obj.set_collision_enabled(collision)

            # add sphere to scene
            self.scene.add(obj)

            # track object
            self._spheres[name] = obj
            return name
        
        def create_cylinder(self, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, color: List[float], collision: bool=True) -> str:
            """
            Spawns a cylinder.
            Must return a unique str identifying the newly spawned object within the engine.
            """
            name = "cylinder" + str(len(self._cylinders))
            prim_path = "/World/" + name

            # create cylinder
            obj = DynamicCylinder(prim_path, position=position, mass=mass, color=self.to_isaac_color(color), radius=radius, orientation=orientation, height=height, name=name)
            obj.set_collision_enabled(collision)

            # add cylinder to scene
            self.scene.add(obj)

            # track object
            self._cylinders[name] = obj
            return name

        def move_base(self, object_id: str, position: np.ndarray, orientation: np.ndarray):
            """
            Moves the base of an object or robot towards the desired position and orientation instantaneously, without physcis calucations.
            """
            raise "Not implemented!"

        ######################################################
        # helper methods (e.g. lines or debug visualization) #
        ######################################################

        def add_aux_line(self, lineFromXYZ: List[float], lineToXYZ: List[float]) -> str:
            """
            Adds a simple line
            """
            raise "Not implemented!"

        def remove_aux_object(self, aux_object_id: str):
            """
            Removes an auxillary object via its int id.
            """
            raise "Not implemented!"

        #################
        # robot methods #
        #################

        def joints_torque_control_velocities(self, robot_id: str, joints_ids: List[int], target_velocities: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
            """
            Sets the velocities of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
            """
            raise "Not implemented!"

        def joints_torque_control_angles(self, robot_id: str, joints_ids: List[int], target_angles: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
            """
            Sets the angles of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
            """
            raise "Not implemented!"

        def set_joints_values(self, robot_id: str, joints_ids: List[int], joints_values: npt.NDArray[np.float32]):
            """
            Same as set_joint_value, but for multiple joints at once.
            """
            # retrieve robot
            robot = self._articulations[robot_id]

            # set joint positions
            robot.set_joint_positions(joints_values, joints_ids)

        def set_joint_value(self, robot_id: str, joint_id: int, joint_value: float):
            """
            Sets the a specific joint to a specific value ignoring phycis, i.e. resulting in instant movement.
            """
            # retrieve robot
            robot = self._articulations[robot_id]

            # set joint state
            robot.set_joint_positions([joint_value], [joint_id])

        def get_joint_value(self, robot_id: str, joint_id: int) -> float:
            """
            Returns the value a single joint is at.
            """
            # retreive robot
            robot = self._articulations[robot_id]

            # retrieve joint
            test = robot.get_joint_positions([joint_id])

            raise "Not implemented!"
        
        def get_link_state(self, robot_id: str, link_id: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns a tuple with position and orientation, both in world frame, of the link in question.
            """
            children = self._articulations[robot_id].prim.GetAllChildren()
            print(children, len(children))

            raise "Not implemented!"
            

        def solve_inverse_kinematics(self, robot_id: str, end_effector_link_id: str, target_position: np.ndarray, target_orientation: Union[np.ndarray, None], max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
            """
            Solves the inverse kinematics problem for the given robot. Returns a vector of joint values.
            If target_orientation is None perform inverse kinematics for position only.
            """
            raise "Not implemented!"

        def get_joints_ids_actuators(self, robot_id: str) -> List[int]:
            """
            This should return a List uniquely identifying (per robot) ints for every joint that is an actuator, e.g. revolute joints but not fixed joints.
            """
            # retreives robot
            robot = self._articulations[robot_id]

            return [i for i in range(robot.num_dof)]
        
        #################
        # ISAAC methods #
        #################

        def get_absolute_asset_path(self, path:str) -> str:
            return Path(self.assets_path).joinpath(path)

        def add_urdf_to_scene(self, prim_path: str) -> Tuple[Articulation, str]:
            """
            A object created as prim is added to the local scene.
            Its given an id and its wrappers, allowing data access, are saved for increased performance
            """
            # create Articulation wrapper, allowing access of joints/values
            obj = Articulation(prim_path)

            # add it to the scene
            self.scene.add(obj)

            # its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
            self.world.reset()

            # track all instances of existing Artriculations
            id = obj.name
            self._articulations[id] = obj
            return obj, id

        def to_isaac_color(self, color: List[float]) -> np.ndarray:
            """
            Transform colour format into format Isaac accepts, ignoring opacity
            """
            return np.array(color[:-1])
        
    
except ImportError:
    # raise error only if Isaac is running
    if is_isaac_running():
        raise

    # Isaac is not enabled, ignore exception
    # Create dummy class to allow referencing
    class IsaacEngine(Engine):
        pass
