from modular_drl_env.task.task import Task
from modular_drl_env.isaac_bridge.bridge import is_isaac_running
from typing import List, TYPE_CHECKING, Tuple, Union, Optional
import numpy as np
import torch
import atexit
from modular_drl_env.world.obstacles.shapes import *

# avoid circular imports
if TYPE_CHECKING:
    from modular_drl_env.robot.robot import Robot
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.sensor import Sensor

try:
    class IsaacTask(Task):
        def __init__(self, asset_path:str, step_size: float, headless:bool=True) -> None:
            super().__init__(asset_path, step_size, headless)
            # isaac imports may only be used after SimulationApp is started (ISAAC uses runtime plugin system)
            from omni.isaac.kit import SimulationApp
            self._simulation = SimulationApp({"headless": headless})

            # make sure simulation was started
            assert self._simulation != None, "Isaac Sim failed to start!"

            # terminate simulation once program exits
            atexit.register(self._simulation.close)

            # retrieve interfaces allowing to access Isaac
            from omni.usd._usd import UsdContext
            self._context: UsdContext = self._simulation.context
            self._app = self._simulation.app

            # create a world, allowing to spawn objects
            from omni.isaac.core import World
            self._world = World(physics_dt=step_size)
            self._scene = self._world.scene
            self._stage = self._world.stage
            
            assert self._world != None, "Isaac world failed to load!"
            assert self._scene != None, "Isaac scene failed to load!"
            assert self._stage != None, "Isaac stage failed to load!"

            # configure urdf importer
            from omni.kit.commands import execute
            result, self._config = execute("URDFCreateImportConfig")
            if not result:
                raise "Failed to create URDF import config"
            
            # set defaults in import config
            self._config.merge_fixed_joints = False
            self._config.convex_decomp = False
            self._config.import_inertia_tensor = True
            self._config.fix_base = True
            self._config.create_physics_scene = True

            # setup physics
            # subscribe to physics contact report event, this callback issued after each simulation step
            from omni.physx import get_physx_simulation_interface
            self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

            # track collisions
            self._collisions: List[Tuple(int, int)] = []

            # configure physics simulation
            from omni.physx.scripts.physicsUtils import UsdPhysics, UsdShade, Gf
            scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(981.0)

            # Configure default floor material
            self._floor_material_path = "/floorMaterial"
            UsdShade.Material.Define(self.stage, self._floor_material_path)
            floor_material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self._floor_material_path))
            floor_material.CreateStaticFrictionAttr().Set(0.0)
            floor_material.CreateDynamicFrictionAttr().Set(0.0)
            floor_material.CreateRestitutionAttr().Set(1.0)

            # Configure default collision material
            self._collision_material_path = "/collisionMaterial"
            UsdShade.Material.Define(self.stage, self._collision_material_path)
            material = UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(self._collision_material_path))
            material.CreateStaticFrictionAttr().Set(0.5)
            material.CreateDynamicFrictionAttr().Set(0.5)
            material.CreateRestitutionAttr().Set(0.9)
            material.CreateDensityAttr().Set(0.001) 

            # setup ground plane
            ground_prim_path = "/World/defaultGroundPlane"
            self._scene.add_default_ground_plane(prim_path=ground_prim_path)

            # add collision to ground plane
            self._add_collision_material(ground_prim_path, self._floor_material_path)

            # track spawned robots/obstacles/sensors
            self._robots_world_path = "/World/Robots/"
            self._obstacles_world_path = "/World/Obstacles/"
            from omni.isaac.core.articulations import ArticulationView
            self._robots = ArticulationView(self._robots_world_path + "*", "Robots")
            self._objects = ArticulationView(f"({self._robots_world_path}|{self._obstacles_world_path})*", "Objects")
            self._sensors = []  # todo: implement sensors

        def set_up(
            self, 
            robots: List[Robot],
            obstacles: List[Obstacle],
            sensors: List[Sensor],
            num_envs: int,
            boundaries: Tuple[float, float, float]
        ) -> Tuple[List[int], List[int], List [int]]:
            # spawn robots
            for i, robot in enumerate(robots):
                # import robot from urdf, creating prim path
                prim_path = self._import_urdf(robot.urdf_path)

                # move imported robot to location of all robots
                prim_path = self._move_prim(prim_path, self._robots_world_path + str(i))

                # configure collision
                if robot.collision:
                    self._add_collision_material(prim_path, self._collision_material_path)
            
            # spawn obstacles
            for i, obstacle in enumerate(obstacles):
                prim_path = self._obstacles_world_path + str(i)

                if isinstance(obstacle, Box):
                    self._create_box(obstacle.args)
                elif isinstance(obstacle, Sphere):
                    self._create_sphere(obstacle.args)
                elif isinstance(obstacle, Cylinder):
                    self._create_cylinder(obstacle.args)
                else:
                    raise f"Obstacle {type(obstacle)} implemented"
            
            # spawn sensors
            for i, sensor in enumerate(sensors):
                raise "Sensors are not implemented"

            # reset world to allow physics object to interact
            self._world.reset()

            # return indices allowing to quickly access robots/obstacles/sensors
            num_robots = len(robots) * num_envs
            num_obstacles = len(obstacles) * num_envs
            num_objects = num_obstacles + num_robots

            assert num_envs == 1, "Multiple environments not implemented!"

            # The regex function will find obstacles before robots -> Obstacles have lower indices than robots 
            return [i for i in range(num_obstacles, num_objects)], [i for i in range(num_obstacles)], [i for i in range(len(sensors))]

        def set_joint_positions(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
            """
            Sets the joint positions of all robots specified in robot_indices to their respective values specified in positions.
            """
            self._robots.set_joint_positions(positions, robot_indices, joint_indices)
        
        def set_joint_position_targets(
            self,
            positions: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
            """
            Sets the joint position targets of all robots specified in robot_indices to their respective values specified in positions.
            """
            self._robots.set_joint_position_targets(positions, robot_indices, joint_indices)

        def set_joint_velocities(
            self,
            velocities: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
            """
            Sets the joint velocities of all robots specified in robot_indices to their respective values specified in velocities.
            """
            self._robots.set_joint_velocities(velocities, robot_indices, joint_indices)
        
        def set_joint_velocity_targets(
            self,
            velocities: Optional[Union[np.ndarray, torch.Tensor]],
            robot_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
            joint_indices: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
        ) -> None:
            """
            Sets the joint velocities targets of all robots specified in robot_indices to their respective values specified in velocities.
            """
            self._robots.set_joint_velocity_targets(velocities, robot_indices, joint_indices)

        def set_local_poses(
            self,
            translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            orientations: Optional[Union[np.ndarray, torch.Tensor]] = None,
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
        ) -> None:
            """
            Sets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
            """
            self._objects.set_local_poses(translations, orientations, indices)

        def get_local_poses(
            self, indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None
        ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
            """
            Gets the local pose, meaning translation and orientation, of all objects (robots and obstacles)
            """
            return self._objects.get_local_poses(indices)

        def get_sensor_data(
            self, 
            indices: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
        ) -> List[List]:
            """
            Gets the sensor data generated by all sensors.
            """
            raise "Not implemented!"

        def get_collisions(self) -> List[Tuple[int, int]]:
            """
            Returns the ids of objects which are colliding. Updated after each step.
            Example: [(1, 2), (1, 3)] -> Object 1 is colliding with object 2 and 3.
            """
            return self._collisions

        def step(self):
            """
            Steps the environment for one timestep
            """
            self._simulation.update()

        def _on_contact_report_event(self, contact_headers, contact_data):
            """
            After each simulation step, ISAAC calles this function. 
            Parameters contain updates about the collision status of spawned objects
            """
            # import required class
            from omni.physx.scripts.physicsUtils import PhysicsSchemaTools

            # clear collisions of previous step
            self._collisions = []

            for contact_header in contact_headers:
                # parse contact information
                contact_type = str(contact_header.type)

                # prim paths of objects with updated collision status
                actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
                actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

                # contact was found
                if 'CONTACT_FOUND' in contact_type or 'CONTACT_PERSIST' in contact_type:
                    self._collisions.append((actor0, actor1)) 

        def _import_urdf(self, urdf_path: str) -> str:
            """
            Loads in a URDF file into the world at position and orientation.
            Must return a unique str identifying the newly spawned object within the engine.
            The is_robot flag determines whether the engine handles this object as a robot (something with movable links/joints) or a simple geometry object (a singular mesh).
            """
            abs_path = self._get_absolute_asset_path(urdf_path)

            # import URDF
            from omni.kit.commands import execute
            success, prim_path = execute("URDFParseAndImportFile", urdf_path=abs_path, import_config=self._config)

            # make sure import succeeded
            assert success, "Failed urdf import of: " + abs_path

            return prim_path

        def _add_collision_material(self, prim_path: str, material_path:str):
            # add material
            from omni.physx.scripts.physicsUtils import add_physics_material_to_prim, PhysxSchema
            add_physics_material_to_prim(self._stage, prim_path, material_path)

            # register contact report api to forward collisions to _on_contact_report_event
            contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(prim_path))
            contactReportAPI.CreateThresholdAttr().Set(200000)

        def _move_prim(self, path_from: str, path_to: str):
            """
            Moves the prim (path of object in simulation) from path_from to path_to.
            Returns the new path of the prim.
            """
            from omni.kit.commands import execute
            success, _ = execute("MovePrim", path_from=path_from, path_to=path_to)

            assert success == True, f"Failed to move prim from {path_from} to {path_to}"

            return path_to

        def _create_box(
            self, 
            prim_path: str,
            position: np.ndarray,
            orientation: np.ndarray,
            mass: float,
            scale: List[float],
            color: List[float],
            collision: bool
            ) -> None:
            from omni.physx.scripts.physicsUtils import add_rigid_box

            # create cube
            add_rigid_box(
                self.stage, prim_path,
                size=to_isaac_vector(scale),
                position=to_isaac_vector(position),
                orientation=to_issac_quat(orientation),
                color=to_isaac_color(color),
                density=mass
            )

            if collision:
                self._add_collision_material(prim_path, self._collision_material_path)

        def _create_sphere(
            self,
            prim_path: str,
            position: np.ndarray,
            mass: float,
            radius: float,
            color: List[float],
            collision: bool
            ) -> None:
            from omni.physx.scripts.physicsUtils import add_rigid_sphere

            # create cube
            add_rigid_sphere(
                self.stage, prim_path,
                radius=radius,
                position=to_isaac_vector(position),
                color=to_isaac_color(color),
                density=mass                
            )

            if collision:
                self._add_collision_material(prim_path, self._collision_material_path)
        
        def _create_cylinder(
            self,
            prim_path: str,
            position: np.ndarray,
            orientation: np.ndarray,
            mass: float,
            radius: float,
            height:float,
            color: List[float],
            collision: bool
        ) -> None:
            from omni.physx.scripts.physicsUtils import add_rigid_cylinder
            add_rigid_cylinder(
                self.stage, prim_path,
                radius=radius,
                height=height,
                position=to_isaac_vector(position),
                orientation=to_isaac_vector(orientation),
                color=to_isaac_color(color),
                density=mass
            )

            if collision:
                self._add_collision_material(prim_path, self._collision_material_path)

        def get_joint_info(self, robot_index: int) -> List[Tuple[str, int]]:
            """
            Returns the joint names and corresponding joint indices of a robot.
            """
            return [(child.GetName(), index) for index, child in enumerate(self._robots.prims[robot_index].GetAllChildren())]
        
    # static utillity functions
    from pxr import Gf
    def to_isaac_vector(vec3: np.ndarray) -> Gf.Vec3f:
        return Gf.Vec3f(list(vec3))

    def to_issac_quat(vec3: np.ndarray) -> Gf.Quatf:
        a, b, c, d = list(vec3)
        return Gf.Quatf(float(a), float(b), float(c), float(d))

    def to_isaac_color(color: List[float]) -> np.ndarray:
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
    class IsaacTask(Task):
        pass