import pybullet as pyb
import numpy as np
from typing import List, Tuple

# this is a small library to abstract away a few things w.r.t. pybullet behind a nicer interface for our use
# what this mainly means that we can use easily understood strings as handles for objects instead of opaque ints
# in other cases the wrapper methods do nothing other than build in conversion of pybullet outputs to numpy arrays

class pybullet_util:
    # these dicts will map pybullet's int ids to strings
    pybullet_object_ids = {}
    gym_env_str_names = {}
    # these will map pybullet's int link ids to their actual names in the source URDF
    pybullet_link_ids = {}
    gym_env_str_link_names = {}
    # these will map pybullet's int joint ids to their actual names in the source URDF
    pybullet_joints_ids = {}
    gym_env_str_joints_names = {}
    # bools to check the current collision state
    collision: bool = False  # bool for whether there is a collision at all
    collisions: List[Tuple[str, str]] = []  # list of tuples of colliding objects

    ##########
    # basics #
    ##########

    @classmethod
    def to_pb(cls, object_id: str) -> int:
        return cls.pybullet_object_ids[object_id]

    @staticmethod
    def init(assets_path: str, display_mode: bool, sim_step: float, gravity: List) -> None:
        pyb.connect(pyb.DIRECT if not display_mode else pyb.GUI)
        if display_mode:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
        pyb.setTimeStep(sim_step)
        pyb.setGravity(*gravity)
        pyb.setAdditionalSearchPath(assets_path)

    @classmethod
    def reset(cls) -> None:
        cls.pybullet_object_ids = {}
        cls.pybullet_link_ids = {}
        cls.gym_env_str_names = {}
        cls.gym_env_str_link_names = {}
        cls.pybullet_joints_ids = {}
        cls.gym_env_str_joints_names = {}
        pyb.resetSimulation()

    @staticmethod
    def close() -> None:
        pyb.disconnect()

    @staticmethod
    def physics_step() -> None:
        pyb.stepSimulation()

    @staticmethod
    def perform_collision_check() -> None:
        pyb.performCollisionDetection()

    @classmethod
    def get_collisions(cls) -> List[Tuple[str, str]]:
        pyb_cols = pyb.getContactPoints()
        ret = []
        for tup in pyb_cols:
            ret.append((cls.gym_env_str_names[tup[1]], cls.gym_env_str_names[tup[2]]))
        cls.collisions = ret
        cls.collision = True if ret else False
    
    ############
    # geometry #
    ############

    @classmethod
    def add_ground_plane(cls, position: np.ndarray) -> str:
        pyb_id = pyb.loadURDF("workspace/plane.urdf", position.tolist())
        cls.gym_env_str_names[pyb_id] = "defaultGroundPlane"
        cls.pybullet_object_ids["defaultGroundPlane"] = pyb.loadURDF("workspace/plane.urdf", position.tolist())
        return "defaultGroundPlane"

    @classmethod
    def load_urdf(cls, urdf_path: str, position: np.ndarray, orientation: np.ndarray, scale: float=1, is_robot: bool=False, fixed_base: bool=True) -> str:
        """
        Loads a URDF file and returns its string id.
        """
        pyb_id = pyb.loadURDF(urdf_path, basePosition=position.tolist(), baseOrientation=orientation.tolist(), useFixedBase=fixed_base, globalScaling=scale)
        
        if is_robot:
            name = "object_" + str(len(cls.pybullet_object_ids)) 
            joints_info = [pyb.getJointInfo(pyb_id, i) for i in range(pyb.getNumJoints(pyb_id))]
            for joint_info in joints_info:
                link_name, link_pyb_id = joint_info[12].decode('UTF-8'), joint_info[0]
                joint_name, joint_pyb_id = joint_info[1].decode('UTF-8'), joint_info[0]
                cls.pybullet_link_ids[(name, link_name)] = link_pyb_id
                cls.gym_env_str_link_names[(pyb_id, link_pyb_id)] = link_name
                cls.pybullet_joints_ids[(name, joint_name)] = joint_pyb_id
                cls.gym_env_str_joints_names[(pyb_id, joint_pyb_id)] = joint_name
        else:
            name = "mesh_" + str(len(cls.pybullet_object_ids)) 

        cls.pybullet_object_ids[name] = pyb_id
        cls.gym_env_str_names[pyb_id] = name

        return name
    
    @classmethod
    def create_box(cls, position: np.ndarray, orientation: np.ndarray, mass: float, halfExtents: List[float]=[1, 1, 1], color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Creates a box and returns its string id.
        """
        name = "box_" + str(len(cls.pybullet_object_ids))
        pyb_id = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents) if collision else -1,
                                    basePosition=position.tolist(),
                                    baseOrientation=orientation.tolist())
        cls.pybullet_object_ids[name] = pyb_id
        cls.gym_env_str_names[pyb_id] = name
        return name
    
    @classmethod
    def create_sphere(cls, position: np.ndarray, mass: float, radius: float, color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Creates a sphere and returns its string id.
        """
        name = "sphere_" + str(len(cls.pybullet_object_ids))
        pyb_id = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=radius, rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=radius) if collision else -1,
                                    basePosition=position.tolist())
        cls.pybullet_object_ids[name] = pyb_id
        cls.gym_env_str_names[pyb_id] = name
        return name
    
    @classmethod
    def create_cylinder(cls, position: np.ndarray, orientation: np.ndarray, mass: float, radius: float, height:float, color: List[float]=[0.5, 0.5, 0.5, 1], collision: bool=True) -> str:
        """
        Creates a cylinder and returns its string id.
        """
        name = "geom_" + str(len(cls.pybullet_object_ids))
        pyb_id = pyb.createMultiBody(baseMass=mass,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_CYLINDER, radius=radius, height=height, rgbaColor=color),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_CYLINDER, radius=radius, height=height) if collision else -1,
                                    basePosition=position.tolist(),
                                    baseOrientation=orientation.tolist())
        cls.pybullet_object_ids[name] = pyb_id
        cls.gym_env_str_names[pyb_id] = name
        return name
    
    @classmethod
    def remove_object(cls, object_id: str) -> None:
        """
        Removes an object from the simulation via its string id.
        """
        pyb.removeBody(cls.pybullet_object_ids[object_id])
        # del cls._pybullet_object_ids[object_id]   # leave this commented out! 
        # we will actually not delete the entry from the dic to prevent the case where, if we spawn something else afterwards,
        # it could potentially lead to double ids, given that we count up for each new object

    @staticmethod
    def draw_lines(starts: List[List[float]], ends: List[List[float]], colors: List[List[float]]) -> List[int]:
        ids = []
        for idx in range(len(starts)):
            line_id = pyb.addUserDebugLine(starts[idx], ends[idx], colors[idx])
            ids.append(line_id)
        return ids
    
    @staticmethod
    def delete_lines(line_ids) -> None:
        for id in line_ids:
            pyb.removeUserDebugItem(id)

    ######################
    # states and control #
    ######################

    @classmethod
    def get_controllable_joint_ids(cls, robot_id: str) -> List[Tuple[str, int]]:
        """
        Returns a list of tuples with string name of joint and joint type for all non-fixed joints.
        """
        pyb_id = cls.pybullet_object_ids[robot_id]

        ret = []
        joints_info = [pyb.getJointInfo(pyb_id, i) for i in range(pyb.getNumJoints(pyb_id))]
        for joint_info in joints_info:
            if joint_info[2] != pyb.JOINT_FIXED:
                ret.append((cls.gym_env_str_joints_names[pyb_id, joint_info[0]], joint_info[2]))
        return ret

    @classmethod
    def get_base_pos_and_ori(cls, object_id) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns base position and orientation for input object.
        """
        pos, ori = pyb.getBasePositionAndOrientation(cls.pybullet_object_ids[object_id])
        return np.array(pos), np.array(ori)

    @classmethod
    def set_base_pos_and_ori(cls, object_id: str, position: np.ndarray, orientation: np.ndarray) -> None:
        pyb_id = cls.pybullet_object_ids[object_id]
        assert pyb_id is not None, "Unknown object id"
        pyb.resetBasePositionAndOrientation(pyb_id, position.tolist(), orientation.tolist())

    @classmethod
    def get_base_vel(cls, object_id) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns velocity and angular velocity for input object.
        """
        vel, ang_vel = pyb.getBaseVelocity(cls.pybullet_object_ids[object_id])
        return np.array(vel), np.array(ang_vel)
    
    @classmethod
    def set_base_vel(cls, object_id: str, vel: np.ndarray, ang_vel: np.ndarray) -> None:
        pyb_id = cls.pybullet_object_ids[object_id]
        assert pyb_id is not None, "Unknown object id"
        pyb.resetBaseVelocity(pyb_id, vel, ang_vel)

    @classmethod
    def get_joint_state(cls, robot_id: str, joint_id: str) -> Tuple[float, float]:
        """
        Returns position and velocity of input joint.
        """
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_id = cls.pybullet_joints_ids[robot_id, joint_id]
        pos, vel, _, _ = pyb.getJointState(pyb_robot_id, pyb_joint_id)
        return pos, vel
    
    @classmethod
    def get_joint_states(cls, robot_id: str, joint_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns two lists in the order input joint ids: joint positions and velocities.
        """
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_ids = [cls.pybullet_joints_ids[robot_id, joint_id] for joint_id in joint_ids]
        pyb_ret = pyb.getJointStates(pyb_robot_id, pyb_joint_ids)
        return np.array([ele[0] for ele in pyb_ret]), np.array([ele[1] for ele in pyb_ret])
    
    @classmethod
    def set_joint_state(cls, robot_id: str, joint_id: str, position: float, velocity: float=0) -> None:
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_id = cls.pybullet_joints_ids[robot_id, joint_id]
        pyb.resetJointState(pyb_robot_id, pyb_joint_id, position, velocity)

    @classmethod
    def set_joint_states(cls, robot_id: str, joint_ids: str, position: np.ndarray[float], velocity: np.ndarray[float]=None) -> None:
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_ids = [cls.pybullet_joints_ids[robot_id, joint_id] for joint_id in joint_ids]

        pybullet_argument_formating = [[value] for value in position]
        if velocity is not None:
            pybullet_argument_formating_vel = [[value] for value in velocity]
        else:
            pybullet_argument_formating_vel = [[0] for _ in position]

        pyb.resetJointStatesMultiDof(pyb_robot_id, pyb_joint_ids, pybullet_argument_formating, pybullet_argument_formating_vel)

    @classmethod
    def set_joint_targets(cls, 
                          robot_id: str, 
                          joint_ids: str, 
                          position: np.ndarray[float]=None, 
                          velocity: np.ndarray[float]=None,
                          forces: np.ndarray[float]=None,
                          max_velocities: np.ndarray[float]=None) -> None:   
        """
        Sets control targets for a robot's joints.
        Can either control position, velocity or both. If both are given, position will take precedence.
        """     
    
        pyb_kwargs = {}
        pyb_kwargs["jointIndices"] = [cls.pybullet_joints_ids[robot_id, joint_id] for joint_id in joint_ids]
        pyb_kwargs["bodyUniqueId"] = cls.pybullet_object_ids[robot_id]
        if velocity is not None:
            pyb_kwargs["targetVelocities"] = velocity
            pyb_kwargs["controlMode"] = pyb.VELOCITY_CONTROL
        if position is not None:
            pyb_kwargs["targetPositions"] = [[ele] for ele in position]
            pyb_kwargs["controlMode"] = pyb.POSITION_CONTROL  # overwrites the velocity control mode
        if forces is not None:
            pyb_kwargs["forces"] = [[ele] for ele in forces]
        if max_velocities is not None:
            pyb_kwargs["maxVelocities"] = max_velocities
        # the [ele] for ele stuff is because the format needs nested lists for the multi dof joints
        # for some reason, the multi dof array method offers more options (max velocities eg) than the normal one

        pyb.setJointMotorControlMultiDofArray(**pyb_kwargs)

    @classmethod
    def get_link_state(cls, robot_id: str, link_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reports position, orientation, velocity and angular velocity of given link.
        """
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_link_id = cls.pybullet_link_ids[robot_id, link_id]
        link_state_pyb = pyb.getLinkState(pyb_robot_id, pyb_link_id, 1, 1)
        return np.array(link_state_pyb[4]), np.array(link_state_pyb[5]), np.array(link_state_pyb[6]), np.array(link_state_pyb[7])
    
    @classmethod
    def get_link_states(cls, robot_id: str, link_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reports positions, orientations, velocities and angular velocities of given links.
        """
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_link_ids = [cls.pybullet_link_ids[robot_id, link_id] for link_id in link_ids]
        link_states_pyb = pyb.getLinkStates(pyb_robot_id, pyb_link_ids, 1, 1)
        return np.array([ele[4] for ele in link_states_pyb]), np.array([ele[5] for ele in link_states_pyb]), np.array([ele[6] for ele in link_states_pyb]), np.array([ele[7] for ele in link_states_pyb])
    
    @classmethod
    def solve_inverse_kinematics(cls, robot_id: str, link_id: str, target_position: np.ndarray, target_orientation: np.ndarray=None, max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_link_id = cls.pybullet_link_ids[robot_id, link_id]
        joints = pyb.calculateInverseKinematics(
            bodyUniqueId=pyb_robot_id,
            endEffectorLinkIndex=pyb_link_id,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            maxNumIterations=max_iterations,
            residualThreshold=threshold
        )
        return np.float32(joints)
    
    ########
    # misc #
    ########

    @classmethod
    def get_joint_dynamics(cls, robot_id: str, joint_id: str) -> Tuple[float, float, float, float]:
        """
        Returns lower limit, upper limit, force limit and max velocity for given joint.
        """
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_id = cls.pybullet_joints_ids[robot_id, joint_id]
        dyn_info = pyb.getJointInfo(pyb_robot_id, pyb_joint_id)
        return dyn_info[8], dyn_info[9], dyn_info[10], dyn_info[11]  # lower, upper, force, velocity

    @classmethod
    def set_joint_dynamics(cls, robot_id: str, joint_id: str, joint_velocity: float, joint_lower_limit: float, joint_upper_limit: float) -> None:
        pyb_robot_id = cls.pybullet_object_ids[robot_id]
        pyb_joint_id = cls.pybullet_joints_ids[robot_id, joint_id]
        pyb.changeDynamics(pyb_robot_id, pyb_joint_id, maxJointVelocity=joint_velocity, jointLowerLimit=joint_lower_limit, jointUpperLimit=joint_upper_limit)
