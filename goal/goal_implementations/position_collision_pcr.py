from goal.goal import Goal
import numpy as np
from robot.robot import Robot
from gym.spaces import Box
import pybullet as pyb
from scipy.spatial.distance import cdist
from numpy import newaxis as na
import time

class PositionCollisionPCR(Goal):
    """
    This class implements a goal of reaching a certain position while avoiding collisions using a point cloud and robot
    skeleton sensor.
    The reward function follows the logic of the paper "Robot obstacle avoidance system using deep
    reinforcement learning".
    """
    def __init__(self, goal_config):
        super().__init__(goal_config)
        self.robot = goal_config["robot"]

        # set pcr and robot skeleton sensor to make retrieving data easier later on
        for sensor in self.robot.sensors:
            if str(type(sensor)) == "<class 'sensor.camera.camera_implementations.static_point_cloud_camera.StaticPointCloudCamera'>":
                self.pcr_sensor = sensor
            if str(type(sensor)) == "<class 'sensor.positional.robot_skeleton_sensor.RobotSkeletonSensor'>":
                self.robot_skeleton_sensor = sensor
        # set the flags
        self.needs_a_position = True
        self.needs_a_rotation = False

        # set the distance thresholds and the increments for changing them
        self.distance_threshold = goal_config["dist_threshold_start"] if self.train else goal_config[
            "dist_threshold_end"]
        if goal_config["dist_threshold_overwrite"]:  # allows to set a different startpoint from the outside
            self.distance_threshold = goal_config["dist_threshold_overwrite"]
        self.distance_threshold_start = goal_config["dist_threshold_start"]
        self.distance_threshold_end = goal_config["dist_threshold_end"]
        self.distance_threshold_increment_start = goal_config["dist_threshold_increment_start"]
        self.distance_threshold_increment_end = goal_config["dist_threshold_increment_end"]

        # placeholders so that we have access in other methods without doing double work
        self.ep_reward = None
        self.distance = None
        self.position = None
        self.reward_value = 0
        self.collided = False
        self.timeout = False
        self.is_success = False
        self.done = False

        # performance metric name
        self.metric_name = "distance_threshold"

        # obstacle point cloud and robot skeleton
        self.obstacle_points : np.array
        self.points : np.array
        self.robot_skeleton : np.array

        # stuff for debugging
        self.debug = goal_config["debug"]
    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            ret["end_effector_position"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
            ret["target_position"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
            ret["closest_obstacle_points"] = Box(low=-1, high=1, shape=(5, 3), dtype=np.float32)
            return ret
        else:
            return {}

    def on_env_reset(self, success_rate):
        self.timeout = False
        self.is_success = False
        self.collided = False
        self.done = False
        self.ep_reward = 0
        # set the distance threshold according to the success of the training
        # set observations
        self._set_observation()

        if self.train:
            # calculate increment
            ratio_start_end = (self.distance_threshold - self.distance_threshold_end) / (
                        self.distance_threshold_start - self.distance_threshold_end)
            increment = (
                                    self.distance_threshold_increment_start - self.distance_threshold_increment_end) * ratio_start_end + self.distance_threshold_increment_end
            if success_rate > 0.7 and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
            elif success_rate < 0.7 and self.distance_threshold < self.distance_threshold_start:
                self.distance_threshold += increment / 25  # upwards movement should be slower
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end

        return self.metric_name, self.distance_threshold, True, True

    def get_observation(self) -> dict:
        # TODO: implement normalization
        return {"end_effector_position": self.robot.position_rotation_sensor.position,
                "target_position": self.target,
                "closest_obstacle_points": self.obstacle_points}

    def _set_observation(self):
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.target = self.robot.world.position_targets[self.robot.id]
        dif = self.target - self.position
        self.distance = np.linalg.norm(dif)
        self._set_min_distance_to_obstacle_and_closest_points()

    def reward(self, step, action):
        reward = 0

        if not self.collided:
            self.collided = self.robot.world.collision

        # set parameters
        lambda_1 = 1
        lambda_2 = 15
        lambda_3 = 0.06
        k = 8
        d_ref = 0.33

        # set observations
        self._set_observation()
        # set motion change
        a = action  # note that the action is normalized

        # reward for distance to target
        R_E_T = -self.distance

        # reward for distance to obstacle
        R_R_O = -(d_ref / (self.min_distance_to_obstacles + d_ref)) ** k

        # reward for motion change
        R_A = - np.sum(np.square(action))

        # success
        self.is_success = False
        if self.collided:
            self.done = True
            reward += -500
        elif self.distance < self.distance_threshold:
            self.done = True
            self.is_success = True
            reward += 500
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += -100
        else:
            # calculate reward
            reward += lambda_1 * R_E_T + lambda_2 * R_R_O + lambda_3 * R_A

        self.ep_reward += reward
        self.reward_value = reward
        return self.reward_value, self.is_success, self.done, self.timeout, False

    def build_visual_aux(self):
        # build a sphere of distance_threshold size around the target
        self.target = self.robot.world.position_targets[self.robot.id]
        self.visual_aux_obj_id = pyb.createMultiBody(baseMass=0,
                                                     baseVisualShapeIndex=pyb.createVisualShape(
                                                         shapeType=pyb.GEOM_SPHERE, radius=self.distance_threshold,
                                                         rgbaColor=[0, 1, 0, 1]),
                                                     basePosition=self.target)

    def get_data_for_logging(self) -> dict:
        logging_dict = dict()
        logging_dict["reward_" + self.robot.name] = self.reward_value
        logging_dict["min_distance_to_obstacles"] = self.min_distance_to_obstacles
        logging_dict["distance_" + self.robot.name] = self.distance
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold
        logging_dict["ep_reward"] = self.ep_reward
        return logging_dict

    def _set_min_distance_to_obstacle_and_closest_points(self):
        """
        Compute the minimal distances from the robot skeleton to the obstacle point, aswell as the 5 closest obstacle points
        to the robot skeleton
        """
        # number of cloest points to retrieve
        num_of_points = 5
        # Compute minimal euclidean distances from the robot skeleton to the obstacle points
        if self.pcr_sensor.points.shape[0] == 0:
            self.obstacle_points = np.repeat(np.array([0, -3, 0])[na, :], 5, axis=0)
            distances_to_obstacles = cdist(self.robot_skeleton_sensor.robot_skeleton,
                                                           np.array([0, -3, 0])[na, :]).min(axis=1).round(10)
        else:
            # all points in the point cloud
            all_points = self.pcr_sensor.points
            # all points except for the ones for the table
            points_without_table = self.pcr_sensor.points[self.pcr_sensor.segImg != 2]
            # concat the two together
            points = np.concatenate([
                points_without_table,
                all_points
            ], axis=0)

            # set indices of the robot skeleton points that should ignore the table
            sklt_indx_ignore_table = [0, 1, 2, 3, 11, 12]
            # set indices of the robot skeleton points that should consider the table
            sklt_indx_consider_table = [4, 5, 6, 7, 8, 9, 10, 13, 14, 15]

            # compute distances for points that should ignore the table
            distances_without_table = cdist(self.robot_skeleton_sensor.robot_skeleton[sklt_indx_ignore_table, :],
                                       points_without_table).min(axis=0)
            # compute distances for points that should consider the table
            distances_with_table = cdist(self.robot_skeleton_sensor.robot_skeleton[sklt_indx_consider_table, :],
                                       all_points).min(axis=0)

            # concat the two together in the same order as the points
            distances = np.concatenate([
                distances_without_table,
                distances_with_table
            ], axis=0)

            # get n closest obstacle points
            n = num_of_points if len(distances) >= num_of_points else len(distances)
            self.obstacle_points = points[np.argpartition(distances, n - 1)][:n]
            distances_to_obstacles = distances.min().round(10)
        # if we have less than n points just add some fake ones that are of the boundary anyways
        n = num_of_points - self.obstacle_points.shape[0]
        if n > 0:
            self.obstacle_points = np.append(self.obstacle_points, np.repeat(np.array([0, -1, 0])[na, :], n, axis=0), axis=0)

        # display closest points
        if self.debug["closest_points"]:
            pyb.removeAllUserDebugItems()
            for point in self.obstacle_points:
                pyb.addUserDebugLine(point, point + np.array([0,0,0.2]))

        self.min_distance_to_obstacles = distances_to_obstacles.astype(np.float32).min()
