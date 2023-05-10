import numpy as np
import pybullet as pyb
from modular_drl_env.planner.planner import Planner
from modular_drl_env.robot.robot import Robot
from typing import List, Tuple
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
import pybullet_planning as pyb_p

__all__ = [
    "RRT",
    "BiRRT"
]

class Node:
    """
    Helper class, node containing a robot configuration q.
    """

    
    def __init__(self, q, parent) -> None:
        self.q = np.array(q)
        self.parent = parent
        self.delta = 0

class Tree:
    """
    Helper class, tree class of nodes for RRT algorithm.
    """

    def __init__(self, q) -> None:
        self.root = Node(q, None)
        self.nodes = [self.root]

    def add_node(self, q, parent):
        new_node = Node(q, parent)
        self.nodes.append(new_node)
        tmp = new_node.parent
        tmp.delta += 1
        while tmp.parent is not None:
            tmp = tmp.parent
            tmp.delta += 1
        return new_node

    def nearest_neighbor(self, q, control):
        min_dist = np.Inf
        closest = None
        for node in self.nodes:
            dist = np.linalg.norm(np.array(q) - np.array(node.q))
            if dist < min_dist and node.delta < control:
                min_dist = dist
                closest = node
        return closest
    
    def retrace(self, node):
        node_trace = [node.q]
        tmp = node
        while tmp.parent is not None:
            node_trace.append(node.parent.q)
            tmp = tmp.parent
        return node_trace[:-1]

    def print_tree(self):
        for node in self.nodes:
            print("---------------------")
            print(node.q)

class RRT(Planner):
    
    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=10000, goal_bias: float=0.35) -> None:
        super().__init__(robot)

        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias

    def plan(self, q_goal) -> List:
        
        # get positions
        q_start = self.robot.joints_sensor.joints_angles
        pos_start, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
        self.robot.moveto_joints(q_goal, False)
        pos_goal, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
        self.robot.moveto_joints(q_start, False)
        
        # init tree
        tree = Tree(q_start)

        control = 1
        r = np.linalg.norm(pos_start - pos_goal)

        for i in range(self.max_iterations):

            if self.goal_bias > np.random.random():
                q_rand = q_goal
            else:
                val = False
                while not val:
                    q_rand = self._sample()
                    self.robot.moveto_joints(q_rand, False)
                    pos_rand, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
                    if np.linalg.norm(pos_goal - pos_rand) < r:
                        val = True

            nearest_node = tree.nearest_neighbor(q_rand, control)
            self.robot.moveto_joints(nearest_node.q, False)

            connect_q, status = self._connect(nearest_node.q, q_rand)

            if status == 0 or status == 1:
                new_node = tree.add_node(connect_q, nearest_node)
                pos_node, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
                r = np.linalg.norm(pos_goal - pos_node)
                control = 1
                if np.array_equal(connect_q, q_goal):
                    path = tree.retrace(new_node)
                    return list(reversed(path))
            elif status == 2:
                r = r + self.epsilon * 50
                control = 3
        

    def _collision_or_oob(self, q: np.ndarray):
        self.robot.moveto_joints(q, False)

        # first check oob
        pos, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
        out = self.robot.world.out_of_bounds(pos)
        if out:
            return True
        else:
            # only check collision after oob because it's much more expensive computationally
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision
        
    def _sample(self):
        while True:
            sample = np.random.uniform(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, size=len(self.robot.controlled_joints_ids))
            if not self._collision_or_oob(sample):
                break
        return sample
    
    def _connect(self, q_start: np.ndarray, q_end: np.ndarray) -> Tuple[np.ndarray, int]:
        q_cur = q_start
        q_old = q_cur
        # main connect loop
        while True:
            diff = q_end - q_cur
            dist = np.linalg.norm(diff)
            if self.epsilon > dist:
                # if this is the case, we can return the goal since we've reached it
                return q_end, 0  # 0: success signal       
            else:
                q_old = q_cur
                q_cur = q_cur + (self.epsilon / dist) * diff
            # check for collision in q_cur
            col_or_out = self._collision_or_oob(q_cur)
            # if we collided and the last node is the start node, we have immediate failure within one epsilon step
            if col_or_out and np.array_equal(q_old, q_start):
                return q_start, 2  # 2: immediate collision
            # if we collided and did at least one step, we can return the last node that was not in collision
            elif col_or_out and not np.array_equal(q_old, q_start):
                return q_old, 1  # 1: at least one step in the direction of q_end

class BiRRT(RRT):

    def __init__(self, robot: Robot, epsilon: float=5e-2, max_iterations: int=1000) -> None:
        super().__init__(robot)
        # get list of pybullet int ids of obstacles
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
        extend_fn = pyb_p.get_extend_fn(pyb_u.to_pb(self.robot.object_id), self.joint_ids)

        def collision_fn(q, diagnosis=False) -> bool:
            q = np.array(q)
            self.robot.moveto_joints(q, False)
            pyb_u.perform_collision_check()
            pyb_u.get_collisions()
            return pyb_u.collision

        if not pyb_p.check_initial_end(q_start, q_goal, collision_fn):
            return [q_start]
        
        while ret is None and tries < 50:
            ret = pyb_p.birrt(q_start, q_goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=40, resolutions=0.005*np.ones(len(self.joint_ids)))
            tries += 1
            if tries >= 50:
                raise Exception("RRT planner stuck!")
        ret = pyb_p.refine_path(pyb_u.to_pb(self.robot.object_id), self.joint_ids, ret, 200)
        pyb_u.perform_collision_check()
        pyb_u.get_collisions()
        #print(pyb_u.collisions)
        self.robot.moveto_joints(q_start, False)
        return np.array(ret)
    
    
    
