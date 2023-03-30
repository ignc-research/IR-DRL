import pybullet as pyb
from ..engine import Engine
from typing import List, TYPE_CHECKING, Union, Tuple
import numpy as np
import numpy.typing as npt

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

class PybulletEngine(Engine):

    def __init__(self, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: list, assets_path: str) -> None:
        super().__init__("Pybullet", use_physics_sim, display_mode, sim_step, gravity, assets_path)
        disp = pyb.DIRECT if not display_mode else pyb.GUI
        pyb.connect(disp)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
        pyb.setTimeStep(sim_step)
        pyb.setGravity(*gravity)
        pyb.setAdditionalSearchPath(assets_path)   

        # object tracking dicts
        self._robots: dict[str, int] = {}
        self._geometry: dict[str, int] = {}
        self._links: dict[(str, str), int] = {} # Tuple: robot str, link str
        self._aux: dict[str, int] = {}


    ###################
    # general methods #
    ###################

    def step(self):
        if self.use_physics_sim:
            pyb.stepSimulation()
        else:
            pass

    def reset(self):
        pyb.resetSimulation()
        self._robots = dict()
        self._geometry = dict()
        self._links = dict()
        self._aux = dict()

    def perform_collision_check(self, robots: List["Robot"], obstacles: List[str]) -> bool:
        pyb.performCollisionDetection()
        col = False
        # check for each robot with every obstacle
        for robot in robots:
            for obstacle in obstacles:
                if len(pyb.getContactPoints(self._robots[robot.object_id], self._geometry[obstacle])) > 0:
                    col = True 
                    break
            if col:
                break  # this is to immediately break out of the outer loop too once a collision has been found
        # check for each robot with every other one
        if not col:  # skip if another collision was already detected
            for idx, robot in enumerate(robots[:-1]):
                for other_robot in robots[idx+1:]:
                    if len(pyb.getContactPoints(self._robots[robot.object_id], self._robots[other_robot.object_id])) > 0:
                        col = True
                        break
                if col:
                    break  # same as above
        return col
    
    def toggle_rendering(self, toggle: bool):
        """
        Turns on or off rendering. Only has a noticeable effect when running with GUI.
        This is useful as some engines are very slow when spawning new objects while rendering, this can bring major speedups.
        If your engine doesn't support this, simply leave this unchanged.
        """
        if toggle:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
        else:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
    
    ####################
    # geometry methods #
    ####################
    
    def add_ground_plane(self, position: np.ndarray) -> str:
        self._geometry["defaultGroundPlane"] = pyb.loadURDF("workspace/plane.urdf", position.tolist())
        return "defaultGroundPlane"
    
    def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: List[float]=[1, 1, 1], is_robot: bool=False) -> str:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        x, y, z = scale
        if not x==y==z:
            print("[load_urdf] Ignoring uneven scaling dimensions for URDF and using only x instead!")
        scale = x
        pyb_id = pyb.loadURDF(urdf_path, basePosition=position.tolist(), baseOrientation=orientation.tolist(), useFixedBase=True, globalScaling=scale)
        if is_robot:
            name = "robot_" + str(len(self._robots))
            self._robots[name] = pyb_id
            # get link ids ...
            joints_info = [pyb.getJointInfo(pyb_id, i) for i in range(pyb.getNumJoints(pyb_id))]
            # and add the correct info to our link dict
            for joint_info in joints_info:
                link_name, link_pyb_id = joint_info[12].decode('UTF-8'), joint_info[0]
                self._links[(name, link_name)] = link_pyb_id
        else:
            name = "mesh_" + str(len(self._geometry))
            self._geometry[name] = pyb_id
        return name
    
    def create_box(self, position: np.ndarray, orientation: np.ndarray, mass: float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Spawns a box at position and orientation. Half extents are the length of the three dimensions starting from position.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        name = "geom_" + str(len(self._geometry))
        self._geometry[name] = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in scale], rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[x/2 for x in scale]) if collision else -1,
                                    basePosition=position.tolist(),
                                    baseOrientation=orientation.tolist())
        return name

    def create_sphere(self, position: np.ndarray, mass: float, radius: float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Spawns a sphere.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        x, y, z = scale
        if not x==y==z:
            print("[create_sphere] Ignoring uneven scale for sphere and only using x value!")
        scale = x
        name = "geom_" + str(len(self._geometry))
        self._geometry[name] = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=radius * scale, rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=radius * scale) if collision else -1,
                                    basePosition=position.tolist())
        return name

    def create_cylinder(self, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, scale: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Spawns a cylinder.
        Must return a unique str identifying the newly spawned object within the engine.
        """
        x, y, z = scale
        if not x==y:
            print("[create_cylinder] Ignoring uneven scale for radius of cylinder and only using x value!")
        name = "geom_" + str(len(self._geometry))
        self._geometry[name] = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=radius * x, height=height * z, rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=radius * x, height=height * z) if collision else -1,
                                    basePosition=position.tolist(),
                                    baseOrientation=orientation.tolist())
        return name

    def move_base(self, object_id: str, position: np.ndarray, orientation: np.ndarray):
        """
        Moves the base of the robot towards the desired position and orientation instantaneously, without physcis calucations.
        """
        pyb_id = self._robots.get(object_id, self._geometry.get(object_id))  # tries to find id in robots, if not tries to find it in geometry objects, will be None if not there also
        assert pyb_id is not None, "Unknown object id"
        pyb.resetBasePositionAndOrientation(pyb_id, position.tolist(), orientation.tolist())
    
    ######################################################
    # helper methods (e.g. lines or debug visualization) #
    ######################################################

    def add_aux_line(self, lineFromXYZ: List[float], lineToXYZ: List[float], color: List[float]=None) -> str:
        name = "line_" + str(len(self._aux))
        self._aux[name] = pyb.addUserDebugLine(lineFromXYZ, lineToXYZ, color)
        return name
    
    def remove_aux_object(self, aux_object_id):
        pyb.removeUserDebugItem(self._aux[aux_object_id])

    def remove_geom_object(self, object_id):
        pyb.removeBody(self._geometry[object_id])
    
    ##################
    # robot movement #
    ##################

    def joints_torque_control_velocities(self, robot_id: str, joints_ids: List[int], target_velocities: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the velocities of the desired joints for the desired robots to the target values. Forces contains the maximum forces that can be used for this.
        """
        pyb.setJointMotorControlArray(self._robots[robot_id], joints_ids, controlMode=pyb.VELOCITY_CONTROL, targetVelocities=target_velocities.tolist(), forces=forces.tolist())

    def joints_torque_control_angles(self, robot_id: str, joints_ids: List[int], target_angles: npt.NDArray[np.float32], forces: npt.NDArray[np.float32]):
        """
        Sets the angles of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        pyb.setJointMotorControlArray(self._robots[robot_id], joints_ids, controlMode=pyb.POSITION_CONTROL, targetPositions=target_angles.tolist(), forces=forces.tolist())

    def set_joint_value(self, robot_id: str, joint_id: int, joint_value: float):
        """
        Sets the a specific joint to a specific value ignoring phycis, i.e. resulting in instant movement.
        """
        pyb.resetJointState(self._robots[robot_id], joint_id, joint_value)

    def get_joint_value(self, robot_id: str, joint_id: int) -> float:
        """
        Returns the value a single joint is at.
        """
        return pyb.getJointState(self._robots[robot_id], joint_id)[0]
    
    def get_link_state(self, robot_id: str, link_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple with position and orientation, both in world frame, of the link in question.
        """
        link_state = pyb.getLinkState(self._robots[robot_id], self._links[robot_id, link_id], computeForwardKinematics=True)
        return np.array(link_state[4]), np.array(link_state[5])

    def solve_inverse_kinematics(self, robot_id: str, end_effector_link_id: str, target_position: np.ndarray, target_orientation: Union[np.ndarray, None], max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
        """
        Solves the inverse kinematics problem for the given robot. Returns a vector of joint values.
        """ 
        joints = pyb.calculateInverseKinematics(
                bodyUniqueId=self._robots[robot_id],
                endEffectorLinkIndex=self._links[robot_id, end_effector_link_id],
                targetPosition=target_position.tolist(),
                targetOrientation=target_orientation.tolist() if target_orientation is not None else None,
                maxNumIterations=max_iterations,
                residualThreshold=threshold)
        return joints

    def get_joints_ids_actuators(self, robot_id: str) -> List[int]:
        """
        This should return a list uniquely identifying (per robot) strs for every joint that is an actuator, e.g. revolute joints but not fixed joints.
        """
        joints_info = [pyb.getJointInfo(self._robots[robot_id], i) for i in range(pyb.getNumJoints(self._robots[robot_id]))]
        return [j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE]
    
    def get_links_ids(self, robot_id: str) -> List[str]:
        """
        This should return a List of uniquely identifying (per robot) strs for every link that makes up the robot.
        """
        ret_list = []
        for key in self._links:
            if key[0] == robot_id:
                ret_list.append(key[1])
        return ret_list


