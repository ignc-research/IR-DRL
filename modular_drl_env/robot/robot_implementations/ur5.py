from typing import Union, List
import numpy as np
from modular_drl_env.robot.robot import Robot
from modular_drl_env.util.rrt import bi_rrt
from time import process_time

__all__ = [
    'UR5',
    'UR5_RRT',
    'UR5_Gripper',
    'UR5_Gripper_RRT'
]

class UR5(Robot):

    def __init__(self, name: str,
                       id_num: int,
                       world,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[]):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, ik_xyz_delta, ik_rpy_delta, joint_velocities_overwrite, joint_limits_overwrite, controlled_joints)
        self.end_effector_link_id = "ee_link"
        self.base_link_id = "base_link"

        self.urdf_path = "robots/predefined/ur5/urdf/ur5.urdf"  

# NOTE: potentially broken, will get rewrite anyway
class UR5_RRT(UR5):

    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], trajectory_tolerance: float, rrt_config: dict):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, 1, 0, 0)
        self.rrt_config = rrt_config
        self.trajectory_tolerance = trajectory_tolerance

    def build(self):
        super().build()
        self.planned_trajectory = []
        self.current_sub_goal = None


    def get_action_space_dims(self):
        return (6, 6)  # doesn't matter for this robot

    def process_action(self, action: np.ndarray):
        cpu_epoch = process_time()
        # if we are calling this for the first time this episode we need to plan the trajectory
        if not self.planned_trajectory:
            q_start = self.joints_sensor.joints_angles
            # check if joints are available for the target
            if len(self.world.joints_targets) >= (self.mgt_id + 1) and self.world.joints_targets[self.mgt_id] is not None:
                q_goal = self.world.joints_targets[self.mgt_id]
            else:  # generate the joints from inverse kinematics
                goal_xyz = self.world.position_targets[self.mgt_id]
                goal_quat = self.world.rotation_targets[self.mgt_id] if self.world.rotation_targets else None
                q_goal = self._solve_ik(goal_xyz, goal_quat)

            self.planned_trajectory = bi_rrt(q_start=q_start,
                                             q_goal=q_goal,
                                             robot=self,
                                             engine=self.engine,
                                             obstacles_ids=self.world.objects_ids,
                                             max_steps=self.rrt_config["max_steps"],
                                             epsilon=self.rrt_config["epsilon"],
                                             goal_bias=self.rrt_config["goal_bias"]
                                             )
            
            if not self.planned_trajectory:
                raise Exception("RRT planning did not find solution after " + str(self.rrt_config["max_steps"]) + " steps!")
            self.current_sub_goal = self.planned_trajectory.pop(0)
            # reset robot after planning
            self.moveto_joints(q_start, False)
        
        # check if we're close enough that we consider the current sub goal fulfilled
        current_config = self.joints_sensor.joints_angles
        config_space_distance = np.linalg.norm(current_config - self.current_sub_goal)
        if config_space_distance < self.trajectory_tolerance:
            self.current_sub_goal = self.planned_trajectory.pop(0)

        
        # transform the current sub goal into a vector between -1 and 1, using the joint limits
        sub_goal_normalized = ((self.current_sub_goal - self.joints_limits_lower) * 2 / self.joints_range) - np.ones(6)
        # now we can use that to call the process_action method, which we use in joint mode (control mode 1 per the init of this class) to achieve the sub goal
        super().process_action(sub_goal_normalized)
        return process_time() - cpu_epoch
    
class UR5_Gripper(UR5):
    def __init__(self, name: str,
                       id_num: int,
                       world,
                       sim_step: float,
                       use_physics_sim: bool,
                       base_position: Union[list, np.ndarray], 
                       base_orientation: Union[list, np.ndarray], 
                       resting_angles: Union[list, np.ndarray], 
                       control_mode: Union[int, str], 
                       ik_xyz_delta: float=0.005,
                       ik_rpy_delta: float=0.005,
                       joint_velocities_overwrite: Union[float, List]=1,
                       joint_limits_overwrite: Union[float, List]=1,
                       controlled_joints: list=[]):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, control_mode, ik_xyz_delta, ik_rpy_delta, joint_velocities_overwrite, joint_limits_overwrite, controlled_joints)

        self.urdf_path = "robots/predefined/ur5/urdf/ur5_with_gripper.urdf"

    # TODO: implement interactivity for the gripper

class UR5_Gripper_RRT(UR5_RRT):
    def __init__(self, name: str, id_num: int, world, sim_step: float, use_physics_sim: bool, base_position: Union[list, np.ndarray], base_orientation: Union[list, np.ndarray], resting_angles: Union[list, np.ndarray], trajectory_tolerance: float, rrt_config: dict):
        super().__init__(name, id_num, world, sim_step, use_physics_sim, base_position, base_orientation, resting_angles, trajectory_tolerance, rrt_config)

        self.urdf_path = "robots/predefined/ur5/urdf/ur5_with_gripper.urdf"