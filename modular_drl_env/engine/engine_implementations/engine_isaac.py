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
    simulation = SimulationApp({"headless": False})
    # terminate simulation once program exits
    atexit.register(simulation.close)

    from omni.kit.commands import execute
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts.physicsUtils import *
    from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema
    from omni.usd._usd import UsdContext
    from pxr.Usd import Prim

    def to_isaac_color(color: List[float]) -> np.ndarray:
        """
        Transform colour format into format Isaac accepts, ignoring opacity
        """
        return np.array(color[:-1])

    def to_isaac_vector(vec3: np.ndarray) -> Gf.Vec3f:
        return Gf.Vec3f(list(vec3))

    def to_issac_quat(vec3: np.ndarray) -> Gf.Quatf:
        a, b, c, d = list(vec3)
        return Gf.Quatf(float(a), float(b), float(c), float(d))

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
            self._config.fix_base = True
            self._config.create_physics_scene = True
            
            # track amount of objects spawned to allow generation unique prim paths
            self.spawned_objects = 0

            # Tracks which id corresponds with which spawned modifyable object
            self._articulations: dict[str, Articulation] = {}

            # subscribe to physics contact report event, this callback issued after each simulation step
            self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

            # configure physics simulation
            scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(981.0)

            # Configure default floor material
            self.floor_material_path = "/floorMaterial"
            UsdShade.Material.Define(self.stage, self.floor_material_path)
            floor_material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self.floor_material_path))
            floor_material.CreateStaticFrictionAttr().Set(0.0)
            floor_material.CreateDynamicFrictionAttr().Set(0.0)
            floor_material.CreateRestitutionAttr().Set(1.0)

            # Configure default collision material
            self.collision_material_path = "/collisionMaterial"
            UsdShade.Material.Define(self.stage, self.collision_material_path)
            material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self.collision_material_path))
            material.CreateStaticFrictionAttr().Set(0.5)
            material.CreateDynamicFrictionAttr().Set(0.5)
            material.CreateRestitutionAttr().Set(0.9)
            material.CreateDensityAttr().Set(0.001) 

            # tracks the ids of objects which are currently colliding
            self.collisions: List[List[str, str]] = []           

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

            # gets the ids of parts all robots are made out of
            link_ids = [self.get_links_ids(robot.object_id) for robot in robots]

            # checks if any robots are colliding
            for i in range(len(robots)):
                robot_part_ids = link_ids[i]
                # check each pair of robots for collision exactly one time
                for j in range(i + 1, len(robots)):
                    if self.are_colliding(robot_part_ids, link_ids[j]):
                        return True
                
                # checks if this robot is colliding with any obstacle
                if self.are_colliding(robot_part_ids, obstacles):
                    return True
            
            # no collisions were detected
            return False            

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
            prim_path = "/" + name

            add_quad_plane(self.stage, prim_path, 'Z', 750, to_isaac_vector(position), Gf.Vec3f(0.5))

            # add collision
            self.add_collision_material(prim_path, self.floor_material_path)

            # return prim_path as id
            return prim_path
        

        def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: List[float]=[1, 1, 1], is_robot: bool=False) -> str:
            """
            Loads in a URDF file into the world at position and orientation.
            Must return a unique str identifying the newly spawned object within the engine.
            The is_robot flag determines whether the engine handles this object as a robot (something with movable links/joints) or a simple geometry object (a singular mesh).
            """

            # get absolute path to urdf
            abs_path = self.get_absolute_asset_path(urdf_path)

            # import URDF
            from omni.kit.commands import execute
            success, prim_path = execute("URDFParseAndImportFile", urdf_path=abs_path, import_config=self._config)
            
            # make sure import succeeded
            assert success, "Failed urdf import of: " + abs_path

            # create wrapper allowing to modify object
            obj = Articulation(prim_path)
            self.scene.add(obj)
            
            # set position, orientation, scale of loaded obj
            obj.set_world_pose(position, orientation)
            obj.set_local_scale(scale)

            # add collision
            self.add_collision_material(prim_path, self.collision_material_path)

            # track object (prim path will always be unique if created by import command)
            self._articulations[prim_path] = obj

            # its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
            self.world.reset() # todo: Must be called after all Articulations were created

            return prim_path
            

        def create_box(self, position: np.ndarray, orientation: np.ndarray, mass: float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
            """
            Spawns a box at position and orientation. Half extents are the length of the three dimensions starting from position.
            Must return a unique str identifying the newly spawned object within the engine.
            """

            # generate unique prim path
            name = "cube" + self.incrementObjectId()
            prim_path = "/" + name

            # create cube
            add_rigid_box(self.stage, prim_path, size=scale, position=to_isaac_vector(position), orientation=to_issac_quat(orientation), color=to_isaac_color(color), density=mass)

            # add collision
            if collision:
                self.add_collision_material(prim_path, self.collision_material_path)

            return prim_path

        def create_sphere(self, position: np.ndarray, mass: float, radius: float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
            """
            Spawns a sphere.
            Must return a unique str identifying the newly spawned object within the engine.
            """

            name = "sphere" + self.incrementObjectId()
            prim_path = "/" + name

            # create sphere
            add_rigid_sphere(self.stage, prim_path, radius, to_isaac_vector(position), color=to_isaac_color(color), density=mass)
            
            # add collision
            if collision:
                self.add_collision_material(prim_path, self.collision_material_path)

            return prim_path
        
        def create_cylinder(self, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
            """
            Spawns a cylinder.
            Must return a unique str identifying the newly spawned object within the engine.
            """
            name = "cylinder" + self.incrementObjectId()
            prim_path = "/World/" + name

            # create cylinder
            add_rigid_cylinder(self.stage, prim_path, radius, height, position=to_isaac_vector(position), orientation=to_isaac_vector(orientation), color=to_isaac_color(color), density=mass)
            
            # add collision
            if collision:
                self.add_collision_material(prim_path)

            return prim_path

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
            return robot.get_joint_positions([joint_id])
            
        
        def get_link_state(self, robot_id: str, link_id: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns a tuple with position and orientation, both in world frame, of the link in question.
            """

            # check if object has been retrieved previously
            obj = self._articulations.get((robot_id, link_id))

            # component was accessed before, retrieve its pose
            if obj is not None:
                return obj.get_world_pose()

            # get all children
            children: List[Prim] = self._articulations[robot_id].prim.GetAllChildren()

            # find the child with matching name
            for child in children:
                if(child.GetName() != link_id):
                    continue
                
                path = child.GetPrimPath()

                obj = Articulation(path)
                self._articulations[(robot_id, link_id)] = obj 

                return obj.get_world_pose()

            raise f"Component of robot {robot_id} with id {link_id} wasn't found!"
            

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

        def get_links_ids(self, robot_id: str) -> List[str]:
            """
            This should return a List of uniquely identifying (per robot) strs for every link that makes up the robot.
            """
            return [child.GetName() for child in self._articulations[robot_id].prim.GetAllChildren()]
        
        #################
        # ISAAC methods #
        #################

        def get_absolute_asset_path(self, path:str) -> str:
            return Path(self.assets_path).joinpath(path)

        def add_collision_material(self, prim_path:str, material_path:str):
            print("Adding collision to:", prim_path)
            # add physics material
            add_physics_material_to_prim(self.stage, self.stage.GetPrimAtPath(Sdf.Path(prim_path)), Sdf.Path(material_path))

            # register contract report
            contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(prim_path))
            contactReportAPI.CreateThresholdAttr().Set(200000)

        def incrementObjectId(self) -> str:
            self.spawned_objects += 1
            return str(self.spawned_objects)

        def _on_contact_report_event(self, contact_headers, contact_data):
            """
            After each simulation step, ISAAC calles this function. 
            Parameters contain updates about the collision status of spawned objects
            """

            for contact_header in contact_headers:
                # parse contact information
                contact_type = str(contact_header.type)

                # prim paths of objects with updated collision status
                actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
                actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

                collision = [actor0, actor1]

                # contact was found
                if 'CONTACT_FOUND' in contact_type:
                    self.collisions.append(collision)

                # contact was lost
                elif 'CONTACT_LOST' in contact_type:
                    self.collisions = [c for c in self.collisions if not collision]

                # contact persists
                elif 'CONTACT_PERSIST' in contact_type:
                    # objects can spawn in a colliding position. 
                    # No contact will be 'found', it registers as 'persisting'
                    if collision not in self.collisions:
                        self.collisions.append(collision)
        

        def are_colliding(self, paths1: List[str], paths2: List[str]) -> bool:
            """
            Given two lists of prim paths, checks if any elements of paths1 are colliding with any elements of paths2.
            Returns true if any do, otherwise false
            """

            for path1 in paths1:
                for path2 in paths2:
                    if self.is_colliding(path1, path2):
                        return True
            return False


        def is_colliding(self, path1: str, path2: str) -> bool:
            """
            Given two prim paths, returns true if the two are colliding, otherwise false.
            """

            # self.collisions tracks prim_paths of colliding objects.
            # List is maintained by the _on_contact_report_event function
             
            # todo: objects seem to always occour in the same order (a colliding with b, never b with a).
            # Second condition can probably be removed for increased performance, but keeping it ensures collisions are always detected

            if [path1, path2] in self.collisions:
                return True
            if [path2, path1] in self.collisions:
                return True
            return False

    
except ImportError:
    # raise error only if Isaac is running
    if is_isaac_running():
        raise

    # Isaac is not enabled, ignore exception
    # Create dummy class to allow referencing
    class IsaacEngine(Engine):
        pass
