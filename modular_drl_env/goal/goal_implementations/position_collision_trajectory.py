from modular_drl_env.goal.goal import Goal
import numpy as np
from modular_drl_env.robot.robot import Robot
from gym.spaces import Box, MultiBinary
from modular_drl_env.util.quaternion_util import quaternion_similarity
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.sensor.sensor_implementations.positional.obstacle_sensor import ObstacleSensor, ObstacleAbsoluteSensor
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
                reward_too_close: float=0.1,
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
            if type(sensor) == ObstacleSensor or type(sensor) == ObstacleAbsoluteSensor:
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
        self.current_position = None
        self.previous_position = None
        self.min_distance_to_trajectory = 0
        self.target_joints = None
        self.final = False
        self.joints_position_buffer = []
        self.joints_position_buffer_size = joints_position_buffer_size
        self.previous_action = None

        self.sample_radius = 0.03

        # small lambda function for proximity reward
        a = (-1 - (-15) * np.exp(self.d_min)) / (1 - np.exp(self.d_min))
        b = -1 - a
        self.proximity_reward = lambda x: a * np.exp(-(x - self.d_min)) + b
        self.proximity_reward = lambda x: (1/((1 - self.d_min) + x))**18

        # performance metric name
        self.metric_names = ["angle_threshold"]

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name + "_JointsTarget"] = Box(low=-1, high=1, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_JointsDelta"] = Box(low=-1, high=1, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_MinDist"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            else:
                ret[self.output_name + "_JointsTarget"] = Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_JointsDelta"] = Box(low=self.robot.joints_limits_lower, high=self.robot.joints_limits_upper, shape=(len(self.robot.controlled_joints_ids),), dtype=np.float32)
                ret[self.output_name + "_MinDist"] = Box(low=0, high=self.obst_sensor.max_distance, shape=(1,), dtype=np.float32)
            ret[self.output_name + "_final"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            return ret
        else:
            return {}
        
    def get_observation(self) -> dict:
        self.previous_position = self.current_position
        self.current_position = self.robot.position_rotation_sensor.position
        self.current_joints = self.robot.joints_sensor.joints_angles
        if self.trajectory is not None:
            self.waypoint_joints = self.trajectory[self.trajectory_idx]
            if self.robot.control_mode == 3:
                self.robot.control_target = self.waypoint_joints
            self.final = self.trajectory_idx / (len(self.trajectory) - 1)
        else:
            self.waypoint_joints = np.zeros((len(self.current_joints),))
            self.final = 0
        self.target_joints = self.robot.world.joints_targets[self.robot.mgt_id]
        
        ret = dict()
        if self.normalize_observations:
            # normalize TODO
            ret[self.output_name + "_JointsTarget"] = np.multiply(self.normalizing_constant_a_obs, self.waypoint_joints) + self.normalizing_constant_b_obs
            ret[self.output_name + "_JointsDelta"] = np.multiply(self.normalizing_constant_a_obs, self.waypoint_joints - self.current_joints) + self.normalizing_constant_b_obs
            ret[self.output_name + "_MinDist"] = [self.obst_sensor.min_dist / self.obst_sensor.max_distance]
        else:
            ret[self.output_name + "_JointsTarget"] = self.waypoint_joints
            ret[self.output_name + "_JointsDelta"] = self.waypoint_joints - self.current_joints
            ret[self.output_name + "_MinDist"] = [self.obst_sensor.min_dist]
        ret[self.output_name + "_final"] = [self.final]
        return ret
    
    def reward(self, step, action):

        if self.trajectory is None:
            return 0, False, True, False, False

        if self.previous_action is None:
            self.previous_action = action

        # check collision status
        self.collided = pyb_u.collision

        # reward for collision
        if self.collided:
            self.done = True
            self.is_success = False
            reward = -30
        # reward for goal reached
        elif self.final == 1 and np.all(np.absolute(self.current_joints - self.waypoint_joints) < self.angle_threshold):
            reward = 75
            self.done = True
            self.is_success = True
        else:
            goal_distance_reward = -0.01 * np.linalg.norm(self.trajectory_xyz[-1] - self.current_position)
            
            action_size_reward = -np.linalg.norm(action)

            # reward for completing part of the trajectory
            trajectory_state_reward = (self.trajectory_idx / (len(self.trajectory) - 1)) * 2.5
            # penalty for not moving forward on the trajectory
            if self.trajectory_idx_prev == self.trajectory_idx:
                moving_forward_reward = -1.5
            else:
                moving_forward_reward = 0
            # penalty for not moving in general
            if np.linalg.norm(self.current_position - self.previous_position) < 0.0015:
                moving_forward_cart_reward = -0.15
            else:
                moving_forward_cart_reward = 0
            #print("distance", self.obst_sensor.min_dist)
            # reward for safe distance
            if self.obst_sensor.min_dist < self.d_min * 2:
                proximity_reward = self.proximity_reward(self.obst_sensor.min_dist)
            else:
                proximity_reward = 0
            # reward for being close to the next trajectory target
            delta_joints = self.waypoint_joints - self.current_joints
            waypoint_reward = -0.5 * np.sum(np.abs(delta_joints))
            trajectory_reward = -0.1 * self.min_distance_to_trajectory
            
            reward = action_size_reward + trajectory_state_reward + moving_forward_reward + \
                            moving_forward_cart_reward + waypoint_reward + trajectory_reward
            
            # weight all the trajectory related rewards down when near an obstacle
            if self.obst_sensor.min_dist < self.d_min:
                reward *= (self.d_min - self.obst_sensor.min_dist)**2
                reward += 5 * goal_distance_reward
            else:
                reward += goal_distance_reward
            # add proximity reward on top
            reward += proximity_reward
            
            if step > self.max_steps:
                self.done = True
                self.is_success = False
                self.timeout = True

            # print("action_size", action_size_reward)
            # print("trajectory_state", trajectory_state_reward)
            # print("moving_forward", moving_forward_reward)
            # print("moving_forward_cart", moving_forward_cart_reward)
            # print("proximity", proximity_reward)
            # print("waypoint", waypoint_reward)
            # print("trajectory", trajectory_reward)
            
        self.reward_value = reward
        
        

        # remember action for next call
        self.previous_action = action

        # determine which waypoint will be going into the observation space for the next step
        # method: in a sphere around the robot's end effector, pick the one waypoint that is both
        # a) the most far away from the end effector while stile in the sphere and
        # b) the closest to the end waypoint along the trajectory
        ee_position = self.robot.position_rotation_sensor.position
        distances_to_ee = np.linalg.norm(self.trajectory_xyz -  ee_position, axis=1)
        self.min_distance_to_trajectory = min(distances_to_ee)
        sphere_mask = distances_to_ee < self.sample_radius
        self.trajectory_idx_prev = self.trajectory_idx
        if sphere_mask.any() == False:  # no points within radius, might happen if the agent goes too far off track
            # take the waypoint with the minimum distance to the robot
            min_distance = np.min(distances_to_ee)
            index_of_that_waypoint = np.where(distances_to_ee == min_distance)[0][0]
            self.trajectory_idx = max(self.trajectory_idx, index_of_that_waypoint)  # keep old waypoint if it's further up in the trajectory
        else:
            waylengths_of_trajectory_points_within_the_sphere = self.trajectory_waylenghts[sphere_mask]
            min_waylength_within_sphere = min(waylengths_of_trajectory_points_within_the_sphere)
            index_of_that_waypoint = np.where(self.trajectory_waylenghts == min_waylength_within_sphere)[0][0]
            self.trajectory_idx = max(self.trajectory_idx, index_of_that_waypoint)  # we use the max here to make sure that the we don't go back in the trajectory

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
        self.current_position = self.robot.position_rotation_sensor.position
        self.previous_position = self.current_position
        self.target_joints = self.robot.world.joints_targets[self.robot.mgt_id]
        self.trajectory_idx = 0
        self.trajectory_idx_prev = 0

        # plan new trajectory      
        self.trajectory = self.planner.plan(self.target_joints, self.robot.world.active_objects)
        
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
        
        # move back to start position (just to be safe in case a planning method leaves the robot somewhere else)    
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
            n_draw_points = min(150, int(0.1 * n_waypoints))
            indices_to_drop = np.random.choice(len(self.trajectory), n_draw_points, replace=False)
            keep_mask = np.zeros(len(self.trajectory), dtype=bool)
            keep_mask[indices_to_drop] = True
            keep_mask[-1] = True
            for waypoint in self.trajectory[keep_mask]:
                waypoint = np.array(waypoint)
                self.robot.moveto_joints(waypoint, False)
                goal_pos, goal_rot, _, _ = pyb_u.get_link_state(self.robot.object_id, self.robot.end_effector_link_id)
                self.robot.moveto_joints(self.current_joints, False)
                color = [0, 1, 0, 0.7] if np.array_equal(waypoint, self.target_joints) else [1, 1, 0, 0.7]
                goal_cylinder = pyb_u.create_cylinder(goal_pos, goal_rot, 0, radius=0.0125, height=0.06, color=color, collision=False)
                self.aux_object_ids.append(goal_cylinder)
        self.robot.moveto_joints(self.current_joints, False)
