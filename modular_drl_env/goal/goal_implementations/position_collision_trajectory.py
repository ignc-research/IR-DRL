from modular_drl_env.goal.goal import Goal
import numpy as np
from modular_drl_env.robot.robot import Robot
from gym.spaces import Box, MultiBinary
from modular_drl_env.util.quaternion_util import quaternion_similarity
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.sensor.sensor_implementations.positional.obstacle_sensor import ObstacleSensor
from modular_drl_env.planner.planner_implementations.rrt import BiRRT, RRT

#TODO: 
# 1. Implement Distance Sensor
# 2. planner class + env reset

class PositionCollisionTrajectoryGoal(Goal):
    """
    This implements a robot goal of following a pre-calculated joint tracjetory while avoiding obstacles
    that might move into its way.
    """

    def __init__(self, 
                robot: Robot, 
                normalize_rewards: bool, 
                normalize_observations: bool, 
                train: bool, 
                add_to_logging: bool, 
                max_steps: int, 
                continue_after_success: bool = False,
                angle_threshold_start=3e-1,
                angle_threshold_end=1e-2,
                angle_threshold_increment_start=1e-2,
                angle_threshold_increment_end=1e-3,
                angle_threshold_overwrite: float=None,
                angle_threshold_change: float=0.8,
                link_reward_weight: list=[1, 1, 1, 1, 1],
                reward_too_close: float=0.225,
                a_dot_threshold: float=0.2,
                w_dist: float=-0.002,
                w_delta: float=0.001,
                w_a: float=-0.0015,
                w_a_dot: float=-0.0004,
                w_waypoint: float=0.05,
                w_collision: float=-0.3,
                joints_position_buffer_size: int=10):

        super().__init__(robot, normalize_rewards, normalize_observations, train, True, add_to_logging, max_steps, continue_after_success)

        # check if necessary sensor is there
        self.obst_sensor = None
        for sensor in self.robot.sensors:
            if type(sensor) == ObstacleSensor:
                self.obst_sensor = sensor
                break
        if self.obst_sensor is None:
            raise Exception("This goal type needs an obstacle sensor to be present for its robot!")

        # set output name for observation space
        self.output_name = "PositionTrajectoryGoal_" + self.robot.name

        # goal type flag
        self.needs_a_joints_position = True
        

        # angle threshold
        self.angle_threshold = angle_threshold_start if self.train else angle_threshold_end
        if angle_threshold_overwrite:  # allows to set a different startpoint from the outside
            self.angle_threshold = angle_threshold_overwrite
        self.angle_threshold_start = angle_threshold_start
        self.angle_threshold_end = angle_threshold_end
        self.angle_threshold_increment_start = angle_threshold_increment_start
        self.angle_threshold_increment_end = angle_threshold_increment_end
        self.angle_threshold_change = angle_threshold_change

        # reward per link weighting
        self.link_reward_weight = link_reward_weight

        # reward function parameters
        self.d_min = reward_too_close
        self.a_dot_threshold = a_dot_threshold
        self.w_dist = w_dist
        self.w_delta = w_delta
        self.w_a = w_a
        self.w_a_dot = w_a_dot
        self.w_reached = w_waypoint
        self.w_collision = w_collision

        # normalizing parameters
        self.normalizing_constant_a_obs = np.zeros(len(self.robot.controlled_joints_ids))
        self.normalizing_constant_b_obs = np.zeros(len(self.robot.controlled_joints_ids))

        self.normalizing_constant_a_obs = 2 / self.robot.joints_range
        self.normalizing_constant_b_obs = np.ones(len(self.robot.joints_range)) - np.multiply(self.normalizing_constant_a_obs, self.robot.joints_limits_upper)

        # init planner # TODO
        #self.planner = RRT(self.robot)
        self.planner = BiRRT(robot)
        self.trajectory = np.zeros((1, len(self.robot.controlled_joints_ids)))
        self.trajectory_idx = 0

        # placeholders so that we have access in other methods without doing double work
        self.reward_value = 0
        self.collided = False
        self.timeout = False
        self.out_of_bounds = False
        self.is_success = False
        self.done = False
        self.current_joints = None
        self.target_joints = None
        self.final = False
        self.joints_position_buffer = []
        self.joints_position_buffer_size = joints_position_buffer_size
        self.previous_action = None

        self.sample_radius = 0.05

        # performance metric name
        self.metric_names = ["angle_threshold"]

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name + "_JointsTarget"] = Box(low=-1, high=1, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_JointsDelta"] = Box(low=-1, high=1, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
            else:
                ret[self.output_name + "_JointsTarget"] = Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_JointsDelta"] = Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
            ret[self.output_name + "_final"] = Box(low=0, high=1, shape=(1,), dtype=int)
            return ret
        else:
            return {}
        
    def get_observation(self) -> dict:
        self.current_joints = self.robot.joints_sensor.joints_angles
        if self.trajectory is not None:
            self.waypoint_joints = self.trajectory[self.trajectory_idx]
            if self.robot.control_mode == 3:
                self.robot.control_target = self.waypoint_joints
            self.final = np.array_equal(self.waypoint_joints, self.trajectory[-1])
        else:
            self.waypoint_joints = np.zeros((len(self.current_joints),))
            self.final = False
        self.target_joints = self.robot.world.joints_targets[self.robot.mgt_id]
        
        ret = dict()
        if self.normalize_observations:
            # normalize TODO
            ret[self.output_name + "_JointsTarget"] = np.multiply(self.normalizing_constant_a_obs, self.waypoint_joints) + self.normalizing_constant_b_obs
            ret[self.output_name + "_JointsDelta"] = np.multiply(self.normalizing_constant_a_obs, self.waypoint_joints - self.current_joints) + self.normalizing_constant_b_obs
        else:
            ret[self.output_name + "_JointsTarget"] = self.waypoint_joints
            ret[self.output_name + "_JointsDelta"] = self.waypoint_joints - self.current_joints
        ret[self.output_name + "_final"] = [int(self.final)]
        return ret
    
    def reward(self, step, action):
        
        if step == 100:
            self.robot.moveto_joints(self.trajectory[-250], False)
            self.robot.joints_sensor.update(0)
            self.robot.position_rotation_sensor.update(0)

        if self.trajectory is None:
            return 0, False, True, False, False

        if self.previous_action is None:
            self.previous_action = action

        # check collision status
        self.collided = pyb_u.collision

        reward = 0

        # Reward weights
        close_distance_weight = -2.0
        delta_joint_weight = 1.0
        action_usage_weight = 1.5
        rapid_action_weight = -0.2
        collision_weight = -0.5
        target_reached_weight = 0.05

        delta_joints = self.waypoint_joints - self.current_joints
        for i in delta_joints:
            if abs(i) < 0.8:
                reward += delta_joint_weight * (1 - (np.abs(i) / 0.8)) * (1 / 1000) * (1 / len(delta_joints))

        reward += action_usage_weight * (1 - (np.square(action).sum() / len(action))) * (1 / 1000)

        for i in range(len(action)):
            if abs(action[i] - self.previous_action[i]) > 0.4:
                reward += rapid_action_weight * (1 / 1000)

        if self.obst_sensor.min_dist < self.d_min:
            reward += close_distance_weight * (1/1000)
        
        if self.collided:
            self.done = True
            reward = collision_weight
        else:
            if self.final and np.all(np.absolute(self.current_joints - self.waypoint_joints) < self.angle_threshold):
                reward += target_reached_weight
                self.done = True
                self.is_success = True

        if step > self.max_steps:
            self.timeout = True
            self.done = True
        """
        # binary signal for when robot is too close to obstacle
        if self.obst_sensor.min_dist < self.d_min:
            r_dist = 1
        else:
            r_dist = 0

        # reward for staying close to the predefined trajectory
        # joints_delta_sum = (np.absolute(self.waypoint_joints - self.current_joints) * self.link_reward_weight).sum()
        # max_joints_delta_sum = (self.robot.joints_range * self.link_reward_weight).sum()
        # r_delta = 1 - (joints_delta_sum/max_joints_delta_sum)**(1/3)

        qt_qr_delta_sum = np.absolute(self.waypoint_joints - self.current_joints).sum()
        threshhold_t = 10
        r_delta = - (1/threshhold_t) * qt_qr_delta_sum + 1
        r_delta = np.clip(r_delta, 0, 1)
        #TODO: define buffer )0 actions e.g in which self.target_joints[0:5] should be closer to target_waypoint
        # if after buffer, the joints are not closer to waypoint, move on to next joint
        
        # reward for action
        r_a = np.mean(np.square(action)) #action 
        r_a_dot = np.sum(np.where(np.absolute(action - self.previous_action) > self.a_dot_threshold, 1, 0))

        # check end conditions
        if self.collided:
            self.done = True
            r_collision = 1
            r_reached = 0
        else:
            r_collision = 0
            if self.final and np.all(np.absolute(self.current_joints - self.waypoint_joints) < self.angle_threshold):
                r_reached = 1
                self.is_success = True
                self.done = True
            else:
                r_reached = 0

        # calculate reward from all the single components
        reward = self.w_dist * r_dist + \
                 self.w_delta * r_delta + \
                 self.w_a * r_a + \
                 self.w_a_dot * r_a_dot + \
                 self.w_reached * r_reached + \
                 self.w_collision * r_collision
        
        if step > self.max_steps:
            self.timeout = True
            self.done = True
        """
            
        self.reward_value = reward

        # remember action for next call
        self.previous_action = action

        # determine which waypoint will be going into the observation space for the next step
        # method: in a sphere around the robot's end effector, pick the one waypoint that is both
        # a) the most far away from the end effector while stile in the sphere and
        # b) the closest to the end waypoint along the trajectory
        ee_position = self.robot.position_rotation_sensor.position
        distances_to_ee = np.linalg.norm(self.trajectory_xyz -  ee_position, axis=1)
        sphere_mask = distances_to_ee < self.sample_radius
        if sphere_mask.any() == False:  # no points within radius, might happen if the agent goes too far off track
            pass  # trajectory idx stays the same
        else:
            waylengths_of_trajectory_points_within_the_sphere = self.trajectory_waylenghts[sphere_mask]
            min_waylength_within_sphere = min(waylengths_of_trajectory_points_within_the_sphere)
            index_of_that_waypoint = np.where(self.trajectory_waylenghts == min_waylength_within_sphere)[0][0]
            self.trajectory_idx = index_of_that_waypoint

        return self.reward_value, self.is_success, self.done, self.timeout, False  # out of bounds always false, doesn't make sense for preplanned trajectory
            
    def on_env_reset(self, success_rate):
        self.n = 0
        # reset attributes
        self.is_success = False
        self.done = False
        self.timeout = False
        self.collided = False
        self.joints_position_buffer = []
        self.final = False
        self.current_joints = self.robot.joints_sensor.joints_angles
        self.target_joints = self.robot.world.joints_targets[self.robot.mgt_id]
        self.trajectory_idx = 0

        # plan new trajectory      
        if hasattr(self.robot.world, 'active_obstacles'):  # if the class has this attribute this saves some collision checking
            self.trajectory = self.planner.plan(self.target_joints, self.robot.world.active_obstacles)
        else:
            self.trajectory = self.planner.plan(self.target_joints, self.robot.world.obstacle_objects)
        
        # calculate cartesian positions of trajectory waypoints
        self.trajectory_xyz = []
        for waypoint in self.trajectory:
            self.robot.moveto_joints(waypoint, False)
            xyz, _, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
            self.trajectory_xyz.append(xyz)
        self.trajectory_xyz = np.array(self.trajectory_xyz)
        # calculate waylengths
        self.trajectory_waylenghts = np.linalg.norm(self.trajectory_xyz[:-1] - self.trajectory_xyz[1:], axis=1)
        self.trajectory_waylenghts = np.array(list(reversed(np.cumsum(list(reversed(self.trajectory_waylenghts))))) + [0])
        

        
        self.robot.moveto_joints(self.current_joints, False)



        if self.trajectory is not None:
            self.waypoint_joints = self.trajectory[0]
            if self.robot.control_mode == 3:  # joint target control mode
                self.robot.control_target = self.waypoint_joints

        if self.train: 

            # calculate increment
            ratio_start_end = (self.angle_threshold - self.angle_threshold_end) / (self.angle_threshold_start - self.angle_threshold_end)
            increment = (self.angle_threshold_increment_start - self.angle_threshold_increment_end) * ratio_start_end + self.angle_threshold_increment_end
            if success_rate > self.angle_threshold_change and self.angle_threshold > self.angle_threshold_end:
                self.angle_threshold -= increment
            elif success_rate < self.angle_threshold_change and self.angle_threshold < self.angle_threshold_start:
                #self.angle_threshold += increment / 25  # upwards movement should be slower # DISABLED
                pass
            if self.angle_threshold > self.angle_threshold_start:
                self.angle_threshold = self.angle_threshold_start
            if self.angle_threshold < self.angle_threshold_end:
                self.angle_threshold = self.angle_threshold_end

        return [("angle_threshold", self.angle_threshold, True, True)]
    
    def build_visual_aux(self):
        if self.trajectory is not None:
            n_waypoints = len(self.trajectory)
            n_draw_points = min(50, int(0.1 * n_waypoints))
            indices_to_drop = np.random.choice(len(self.trajectory), n_draw_points, replace=False)
            keep_mask = np.zeros(len(self.trajectory), dtype=bool)
            keep_mask[indices_to_drop] = True
            for waypoint in self.trajectory[keep_mask]:
                waypoint = np.array(waypoint)
                self.robot.moveto_joints(waypoint, False)
                goal_pos, goal_rot, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
                self.robot.moveto_joints(self.current_joints, False)
                color = [0, 1, 0, 0.7] if np.array_equal(waypoint, self.target_joints) else [1, 1, 0, 0.7]
                goal_cylinder = pyb_u.create_cylinder(goal_pos, goal_rot, 0, radius=0.0125, height=0.06, color=color, collision=False)
                self.aux_object_ids.append(goal_cylinder)
        self.robot.moveto_joints(self.current_joints, False)

class PositionCollisionTrajectoryGoalCartesian(Goal):
    """
    Goal that gives the robot a trajectory to follow. In default mode, it will try to go to the target in a straight line.
    TODO
    """
