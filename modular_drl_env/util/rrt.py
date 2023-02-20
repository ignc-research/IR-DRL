import numpy as np
import pybullet as pyb

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
    
    def print_tree(self):
        for node in self.nodes:
            print("---------------------")
            print(node.q)

def get_collision_or_out_fn(robot, obstacles_ids):
    """
    Creates a collision/oob function for a Robot robot and
    a list of obstacle ids obstacles_ids.
    """

    def collision_or_oob(q):
        robot.moveto_joints(q, False)

        # first check oob
        pos = np.array(pyb.getLinkState(robot.object_id, robot.end_effector_link_id)[4])
        out = robot.world.out_of_bounds(pos)
        if out:
            return True
        else:
            # only check collision after oob because it's much more expensive computationally
            pyb.performCollisionDetection()

            col = False
            for obst in obstacles_ids:
                if pyb.getContactPoints(robot.object_id, obst):
                    col = True
                    break
            return col

    return collision_or_oob

def get_sample_fn(robot):
    """
    Creates a function for sampling the configuration space of a robot.
    """
    def out(q):
        # we rewrite the out function here because checking every sample for both oob and collision would be bad for performance
        robot.moveto_joints(q, False)
        pos = np.array(pyb.getLinkState(robot.object_id, robot.end_effector_link_id)[4])
        out = robot.world.out_of_bounds(pos)
        return out

    def sample():
        sample = np.random.uniform(low=robot.joints_limits_lower, high=robot.joints_limits_upper, size=len(robot.joints_ids))
        # check if sample is in workspace bounds
        while out(sample):
            sample = np.random.uniform(low=robot.joints_limits_lower, high=robot.joints_limits_upper, size=len(robot.joints_ids))
        return sample
    return sample

def get_connect_fn(collision_or_out, epsilon=1e-3):
    """
    Creates a connect function that can connect a configuration to a tree.
    """
    def connect(tree: Tree, q: np.ndarray, control):
        # get nearest node in tree
        node_near = tree.nearest_neighbor(q, control)
        q_cur = node_near.q
        q_old = q_cur
        # connect loop
        while True:
            # get distance in joint space
            diff = q - q_cur
            dist = np.linalg.norm(diff)
            # take an epsilon step towards the goal joint position
            if epsilon > dist:
                q_cur = q
            else:
                q_old = q_cur
                q_cur = q_cur + (epsilon / dist) * diff
            # if we reached the goal, add to tree and transmit 0 for total success
            if np.array_equal(q_cur, q):
                return tree.add_node(q, node_near), 0
            col_or_out = collision_or_out(q_cur)
            # if we immediately collide within one epsilon, 
            # return the old node and transmit 2 for failure
            if col_or_out and np.array_equal(q_old, node_near.q):
                return node_near, 2
            # if we collided after at least one epsilon step
            # add the last node before collision and transmit 1 for partial success
            elif col_or_out and not np.array_equal(q_old, node_near.q):
                return tree.add_node(q_old, node_near), 1
    return connect

def bi_path(node1, node2, tree1, q_start):
    """
    For a two node objects node1 and node2 that hold the same configuration value q but belong to two different trees,
    this returns a path from the root of tree 1 to the root of tree 2.
    Tree1 is the tree containing the node1, q_start is the start config of the path.
    """

    # find out which node belongs to the tree that is the one growing from the start config
    if np.array_equal(tree1.root.q, q_start):
        nodeA = node1
        nodeB = node2
    else:
        nodeA = node2
        nodeB = node1

    # go from connection to root for both tree
    tmp = nodeA
    a_traj = [nodeA.q]
    while tmp.parent is not None:
        tmp = tmp.parent
        a_traj.append(tmp.q)
    tmp = nodeB
    b_traj = [nodeB.q]
    while tmp.parent is not None:
        tmp = tmp.parent
        b_traj.append(tmp.q)

    return list(reversed(a_traj)) + b_traj

def bi_rrt(q_start, q_goal, robot, obstacles_ids, max_steps, epsilon, goal_bias, visible=False, force_swap=200):
    """
    Bi-RRT algorithm. Returns a valid, collision free trajectory of joint positions
    """

    treeA = Tree(q_start)
    treeB = Tree(q_goal)

    # radii for biased sampling
    rA = np.linalg.norm(np.array(q_goal) - np.array(q_start))
    rB = rA

    # control factors
    controlA, controlB = 1, 1

    # tracking for adaptive swap
    collisionA, triesA, ratioA = 0, 0, 1.
    collisionB, triesB, ratioB = 0, 0, 1.

    # free run, number of times after a force swap of trees that the swap condition will be ignored
    free_runs = 50
    free_run = 0
    force_swap_active = False

    # anchorpoint sampling init
    anchorpoint = np.zeros(len(q_start))
    scale = 1

    # stop pybullet rendering
    if not visible:
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)


    # get collision function
    collision_or_out = get_collision_or_out_fn(robot, obstacles_ids)

    # check goal and start
    if collision_or_out(q_start):
        raise Exception("Start configuration is in collision or out of bounds!")
    if collision_or_out(q_goal):
        raise Exception("Goal configuration is in collision or out of bounds!")

    # get the connect function
    connect = get_connect_fn(collision_or_out, epsilon)

    # get the sampling function
    sample = get_sample_fn(robot)

    # main algorithm
    for i in range(max_steps):

        # sampling
        random = np.random.random()
        # goal bias
        if random > goal_bias:
            # get random sample
            q_rand = sample() * scale + anchorpoint
            if np.linalg.norm(q_rand - treeB.root.q) > rA:
                # resample if it's not within sample bias radius
                continue
        else:
            # try goal node
            q_rand = treeB.root.q

        # attempt to connect to tree
        reached_nodeA, status = connect(treeA, q_rand, controlA)
        # increment connect tries
        triesA += 1

        # anchorpoint sampling handling
        if status == 1 or status == 2:
            # connection unsuccessful, reduce sampling raidus and use last node before collision as anchorpoint
            scale = max(0.05, scale - 0.05)
            anchorpoint = reached_nodeA.q
        if status == 0:
            # conncetion successful, reset anchorpoint and scale
            scale = 1
            anchorpoint = np.zeros(len(q_start))

        # connect attempt results
        if status == 2:   # instant collision
            collisionA += 1
            # make sampling radius larger
            rA = rA + epsilon * 50
            # (temporarily allow nodes from inside the tree to be connected to)
            controlA = 3
        else:  # some node, not necessarily the sampled one, was connected to
            # set new sampling bias radius
            rA = np.linalg.norm(reached_nodeA.q - treeB.root.q)
            # allow again only border nodes
            controlA = 1
            # try to reach the node from the other tree
            reached_nodeB, status = connect(treeB, reached_nodeA.q, controlB)
            if status == 0:  # success, connection between trees established
                # get path from start to finish
                sol = bi_path(reached_nodeA, reached_nodeB, treeA, q_start)#
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                return sol
            elif status == 2:
                collisionB += 1

        # tree swap
        ratioA = collisionA / triesA
        ratioB = 1 if triesB == 0 else collisionB / triesB
        # if the current tree was swapped not by ratio but by force, count up the free runs it gets before swapping is determined by ratio again
        if force_swap_active:
            free_run += 1
            if free_run > free_runs:
                free_run = 0
                force_swap_active = False
        if (ratioB > ratioA and not force_swap_active) or i%force_swap==0:
            if i%force_swap==0:
                force_swap_active = True
            treeA, treeB = treeB, treeA
            collisionA, collisionB = collisionB, collisionA
            triesA, triesB = triesB, triesA
            ratioA, ratioB = ratioB, ratioA
            rA, rB = rB, rA
            controlA, controlB = controlB, controlA
            anchorpoint = np.zeros(len(q_start))
            scale = 1

    pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
    
    return None

def smooth_path(path, epsilon, robot, obstacles_ids):
    """
    Takes a working path and smoothes it by checking if intermediate steps can be skipped.
    Greedy and thus pretty slow.
    """

    collision = get_collision_or_out_fn(robot, obstacles_ids)

    def free(q_start, q_end, epsilon):
        tmp = q_start
        while True:
            dist = np.linalg.norm(q_end - tmp)
            if epsilon > dist:
                return True
            else:
                tmp = tmp + (epsilon/dist) * (q_end - tmp)
                if collision(tmp):
                    return False       
    
    path_smooth = [path[0]]
    cur = 0

    while cur < len(path) - 1:
        for idx, pose in reversed(list(enumerate(path[cur+1:]))):
            if free(path[cur], pose, epsilon):
                path_smooth.append(pose)
                cur = idx+cur
                break
        cur += 1

    return path_smooth