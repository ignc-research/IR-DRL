import numpy as np
import pybullet as pyb
from modular_drl_env.planner.planner import Planner
from modular_drl_env.robot.robot import Robot
from typing import List, Tuple
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet_planning as pyb_p
from time import process_time, time

__all__ = [
    "RRT",
    "BiRRT",
    "RRTStar"
]
class RRT(Planner):
    
    def __init__(self, robot: Robot, epsilon: float=0.0416, max_iterations: int=10000, goal_bias: float=0.35, padding: bool=True) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]
        self.joint_ids_u = self.robot.controlled_joints_ids

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.padding = padding

    def plan(self, q_goal, obstacles) -> List:
        obstacles_pyb = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start, _ = pyb_u.get_joint_states(self.robot.object_id, self.joint_ids_u)
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids, resolutions=self.epsilon*np.ones(len(self.joint_ids)))

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.joint_ids_u)
            for obst in obstacles:
                if obst.seen_by_obstacle_sensor:
                    if pyb.getClosestPoints(pyb_u.to_pb(self.robot.object_id), pyb_u.to_pb(obst.object_id), 0.03):
                        return True
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        t_start = time()
        while ret is None and tries < 5000:
            
            ret = pyb_p.rrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100)
            tries += 1
            if tries >= 5000:
                break
        pyb_u.log_planner_time(self.robot.object_id, time() - t_start)
        if ret is None:
            ret = [q_start, q_goal]
        if self.padding:
            ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False, self.joint_ids_u)
        return np.array(ret)

class BiRRT(Planner):

    def __init__(self, robot: Robot, epsilon: float=0.0416, max_iterations: int=1000, padding: bool=True) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]
        self.joint_ids_u = self.robot.controlled_joints_ids

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.padding = padding

    def plan(self, q_goal, obstacles) -> List:
        obstacles_pyb = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start, _ = pyb_u.get_joint_states(self.robot.object_id, self.joint_ids_u)
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids, resolutions=self.epsilon*np.ones(len(self.joint_ids)))

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.joint_ids_u)
            for obst in obstacles:
                if obst.seen_by_obstacle_sensor:
                    if pyb.getClosestPoints(pyb_u.to_pb(self.robot.object_id), pyb_u.to_pb(obst.object_id), 0.03):
                        return True
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        t_start = time()
        while ret is None and tries < 50:      
            ret = pyb_p.birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, resolution=self.epsilon)       
            #ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, radius=50, informed=False)
            #ret = pyb_p.prm(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
            tries += 1
            if tries >= 50:
                break
        pyb_u.log_planner_time(self.robot.object_id, time() - t_start)
        if ret is None:
            ret = [q_start, q_goal]
        if self.padding:
            ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False, self.joint_ids_u)
        return np.array(ret)
    
class RRTStar(Planner):

    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=1000, padding: bool=True) -> None:
        super().__init__(robot)
        self.joint_ids = [pyb_u.pybullet_joints_ids[self.robot.object_id, joint_id] for joint_id in self.robot.controlled_joints_ids]
        self.joint_ids_u = self.robot.controlled_joints_ids

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.padding = padding

    def plan(self, q_goal, obstacles) -> List:
        obstacles = [pyb_u.to_pb(obstacle.object_id) for obstacle in obstacles]
        q_start, _ = pyb_u.get_joint_states(self.robot.object_id, self.joint_ids_u)
        ret = None
        tries = 0
        sample_fn = pyb_p.get_sample_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        distance_fn = pyb_p.get_distance_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids, resolutions=0.005*np.ones(len(self.joint_ids)))

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False, self.joint_ids_u)
            for obst in obstacles:
                if pyb.getClosestPoints(pyb_u.to_pb(self.robot.object_id), obst, 0.01):
                    return True
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        t_start = time()
        while ret is None and tries < 50:
            ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, 2, max_iterations=100)
            #ret = pyb_p.rrt_star(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, radius=50, informed=False)
            #ret = pyb_p.prm(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn)
            tries += 1
            if tries >= 50:
                break
        pyb_u.log_planner_time(self.robot.object_id, time() - t_start)
        if ret is None:
            ret = [q_start, q_goal]
        if self.padding:
            ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False, self.joint_ids_u)
        return np.array(ret)  
    
