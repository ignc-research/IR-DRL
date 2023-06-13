import numpy as np
import pybullet as pyb
from modular_drl_env.planner.planner import Planner
from modular_drl_env.robot.robot import Robot
from typing import List, Tuple
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet_planning as pyb_p
from time import process_time

__all__ = [
    "RRT",
    "BiRRT",
    "RRTStar"
]
class RRT(Planner):
    
    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=10000, goal_bias: float=0.35) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias

    def plan(self, q_goal, obstacles) -> List:
        obstacles = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start = self.robot.joints_sensor.joints_angles
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.robot.controlled_joints_ids)
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        while ret is None and tries < 50:
            t_start = process_time()
            ret = pyb_p.rrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100)
            
            pyb_u.log_planner_time(self.robot.object_id, process_time() - t_start)
            tries += 1
            if tries >= 50:
                break
        if ret is None:
            ret = [q_start, q_goal]
        ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False)
        return np.array(ret)

class BiRRT(Planner):

    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=1000) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]

        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def plan(self, q_goal, obstacles) -> List:
        obstacles = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start = self.robot.joints_sensor.joints_angles
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids, resolutions=0.05*np.ones(len(self.joint_ids)))

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.robot.controlled_joints_ids)
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        while ret is None and tries < 50:
            t_start = process_time()
            ret = pyb_p.birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100)
            pyb_u.log_planner_time(self.robot.object_id, process_time() - t_start)
            #ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, radius=50, informed=False)
            #ret = pyb_p.prm(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
            tries += 1
            if tries >= 50:
                break
        if ret is None:
            ret = [q_start, q_goal]
        ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False)
        return np.array(ret)
    
class RRTStar(Planner):

    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=1000) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]

        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def plan(self, q_goal, obstacles) -> List:
        obstacles = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start = self.robot.joints_sensor.joints_angles
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids, resolutions=0.005*np.ones(len(self.joint_ids)))

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.robot.controlled_joints_ids)
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        while ret is None and tries < 50:
            t_start = process_time()
            ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, 2, max_iterations=100)
            pyb_u.log_planner_time(self.robot.object_id, process_time() - t_start)
            #ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, radius=50, informed=False)
            #ret = pyb_p.prm(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
            tries += 1
            if tries >= 50:
                break
        if ret is None:
            ret = [q_start, q_goal]
        ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False)
        return np.array(ret)  
    
