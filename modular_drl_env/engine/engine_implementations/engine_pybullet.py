import pybullet as pyb
from ..engine import Engine
from typing import List, TYPE_CHECKING, Union
import numpy as np

# Use type checking to enable tyhe hints and prevent circular imports
if TYPE_CHECKING:
    from modular_drl_env.world.obstacles.obstacle import Obstacle
    from modular_drl_env.robot.robot import Robot

class PybulletEngine(Engine):

    def __init__(self, use_physics_sim: bool, display_mode: bool, sim_step: float, gravity: list, assets_path: str) -> None:
        super().__init__(use_physics_sim, display_mode, sim_step, gravity, assets_path)
        disp = pyb.DIRECT if not display_mode else pyb.GUI
        pyb.connect(disp)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
        pyb.setTimeStep(sim_step)
        pyb.setGravity(*gravity)
        pyb.setAdditionalSearchPath(assets_path)     

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

    def perform_collision_check(self, robots: List["Robot"], obstacles: List[int]) -> bool:
        pyb.performCollisionDetection()
        col = False
        # check for each robot with every obstacle
        for robot in robots:
            for obstacle in obstacles:
                if len(pyb.getContactPoints(robot.object_id, obstacle)) > 0:
                    col = True 
                    break
            if col:
                break  # this is to immediately break out of the outer loop too once a collision has been found
        # check for each robot with every other one
        if not col:  # skip if another collision was already detected
            for idx, robot in enumerate(robots[:-1]):
                for other_robot in robots[idx+1:]:
                    if len(pyb.getContactPoints(robot.object_id, other_robot.object_id)) > 0:
                        col = True
                        break
                if col:
                    break  # same as above
        return col
    
    ####################
    # geometry methods #
    ####################
    
    def add_ground_plane(self, position: np.ndarray):
        return pyb.loadURDF("workspace/plane.urdf", position.tolist())
    
    def load_urdf(self, urdf_path: str, position: np.ndarray, orientation: np.ndarray) -> int:
        """
        Loads in a URDF file into the world at position and orientation.
        Must return a unique int identifying the newly spawned object within the engine.
        """
        return pyb.loadURDF(urdf_path, basePosition=position.tolist(), baseOrientation=orientation.tolist(), useFixedBase=True)
    
    ######################################################
    # helper methods (e.g. lines or debug visualization) #
    ######################################################

    def addUserDebugLine(self, lineFromXYZ: List[float], lineToXYZ: List[float]):
        return pyb.addUserDebugLine(lineFromXYZ, lineToXYZ)
    
    ##################
    # robot movement #
    ##################

    def joints_torque_control_velocities(self, robot_id: int, joints_ids: list[int], target_velocities: np.ndarray[float], forces: np.ndarray[float]):
        """
        Sets the velocities of the desired joints for the desired robots to the target values. Forces contains the maximum forces that can be used for this.
        """
        pyb.setJointMotorControlArray(robot_id, joints_ids, controlMode=pyb.VELOCITY_CONTROL, targetVelocities=target_velocities.tolist(), forces=forces.tolist())

    def joints_torque_control_angles(self, robot_id: int, joints_ids: list[int], target_angles: np.ndarray[float], forces: np.ndarray[float]):
        """
        Sets the angles of the desired joints for the desired robot to the target values using the robot's actuators. Forces contains the maximum forces that can be used for this.
        """
        pyb.setJointMotorControlArray(robot_id, joints_ids, controlMode=pyb.POSITION_CONTROL, targetPositions=target_angles.tolist(), forces=forces.tolist())

    def set_joint_value(self, robot_id: int, joint_id: int, joint_value: float):
        """
        Sets the a specific joint to a specific value ignoring phycis, i.e. resulting in instant movement.
        """
        pyb.resetJointState(robot_id, joint_id, joint_value)

    def solve_inverse_kinematics(self, robot_id: int, end_effector_link_id: int, target_position: np.ndarray, target_orientation: Union[np.ndarray, None], max_iterations: int=100, threshold: float=1e-2) -> np.ndarray:
        """
        Solves the inverse kinematics problem for the given robot. Returns a vector of joint values.
        """ 
        joints = pyb.calculateInverseKinematics(
                bodyUniqueId=robot_id,
                endEffectorLinkIndex=end_effector_link_id,
                targetPosition=target_position.tolist(),
                targetOrientation=target_orientation.tolist() if target_orientation is not None else None,
                maxNumIterations=max_iterations,
                residualThreshold=threshold)
        return joints
    
    def move_base(self, robot_id, position: np.ndarray, orientation: np.ndarray):
        """
        Moves the base of the robot towards the desired position and orientation instantaneously, without physcis calucations.
        """
        pyb.resetBasePositionAndOrientation(robot_id, position.tolist(), orientation.tolist())

    def get_joints_ids_actuators(self, robot_id) -> list[int]:
        """
        This should return a list uniquely identifying (per robot) ints for every joint that is an actuator, e.g. revolute joints but not fixed joints.
        """
        joints_info = [pyb.getJointInfo(robot_id, i) for i in range(pyb.getNumJoints(robot_id))]
        return [j[0] for j in joints_info if j[2] == pyb.JOINT_REVOLUTE]


