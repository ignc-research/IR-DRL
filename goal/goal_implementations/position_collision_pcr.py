from goal.goal import Goal
import numpy as np
from gym.spaces import Box
import pybullet as pyb
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
            if str(type(
                    sensor)) == "<class 'sensor.camera.camera_implementations.static_point_cloud_camera.StaticPointCloudCamera'>":
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
        self.cpu_epoch = None
        self.step = None
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

        # set indices of the robot skeleton points that should ignore the table
        self.sklt_indx_ignore_table = [0, 1, 2, 3, 10, 11, 12]

        # set indices of the robot skeleton points that should consider the table
        self.sklt_indx_consider_table = [4, 5, 6, 7, 8, 9, 13, 14, 15]

        # stuff for debugging
        self.debug = goal_config["debug"]

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            # ret["end_effector_position"] = Box(low=np.array([-1, -1, 1], dtype=np.float32),
            #                                    high=np.array([1, 1, 2], dtype=np.float32),
            #                                    shape=(3,), dtype=np.float32)
            ret["target_position"] = Box(low=np.array([-1, -1, 1], dtype=np.float32),
                                         high=np.array([1, 1, 2], dtype=np.float32),
                                         shape=(3,), dtype=np.float32)
            ret["closest_obstacle_cuboid"] = Box(low=np.array([0, 0, 0, -1, -1, 1]),
                                                 high=np.array([1.5, 1.5, 1, 1, 1, 2]), shape=(6,),
                                                 dtype=np.float32)
            return ret
        else:
            return {}

    def on_env_reset(self, success_rate):
        t = time.time()
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
            if success_rate > 0.65 and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
            elif success_rate < 0.65 and self.distance_threshold < self.distance_threshold_start:
                self.distance_threshold += increment / 25  # upwards movement should be slower
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end

        self.cpu_epoch = time.time() - t
        return self.metric_name, self.distance_threshold, True, True

    def get_observation(self) -> dict:
        # TODO: implement normalization
        return {"target_position": self.target,
                "closest_obstacle_cuboid": self.closest_obstacle_cuboid[6:]}

    def _set_observation(self):
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.target = self.robot.world.position_targets[self.robot.id]
        dif = self.target - self.position
        self.distance = np.linalg.norm(dif)
        self._set_min_distance_to_obstacle_and_closest_points()

    def reward(self, step, action):
        t = time.time()

        reward = 0
        self.step = step
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

        self.cpu_epoch = time.time() - t
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
        logging_dict["goal_cpu_time"] = self.cpu_epoch
        return logging_dict

    def _set_min_distance_to_obstacle_and_closest_points(self):
        """
        Set the closest obstacle cuboid and the shortest distance between robot skeleton and obstacle cuboids.
        """
        # retrieve robot skeleton and obstacle cuboids
        robot_sklt = self.robot_skeleton_sensor.robot_skeleton
        # obstacles: [x_max, x_min, y_max, y_min, z_max, z_min, length, depth, height, x_center, y_center, z_center]
        obstacle_cuboids = self.pcr_sensor.obstacle_cuboids
        obstacles_expanded = self.pcr_sensor.obstacle_cuboids[:, :, na].repeat(robot_sklt.shape[0], axis=2)

        # check relative x position
        is_to_right = robot_sklt[:, 0] > obstacles_expanded[:, 0, :]
        is_to_left = robot_sklt[:, 0] < obstacles_expanded[:, 1, :]
        # check relative y position
        is_infront = robot_sklt[:, 1] > obstacles_expanded[:, 2, :]
        is_behind = robot_sklt[:, 1] < obstacles_expanded[:, 3, :]
        # check relative z position
        is_above = robot_sklt[:, 2] > obstacles_expanded[:, 4, :]
        is_below = robot_sklt[:, 2] < obstacles_expanded[:, 5, :]

        # set array containt the projections; should have shape n_of_obstacles x n_robot_skeleton_points x 3
        robot_sklt_projections = robot_sklt[na, :, :].repeat(obstacles_expanded.shape[0], axis=0)

        # if is_to_right x_projection = x_max; if is_to_left x_projection = x_min
        robot_sklt_projections[:, :, 0] = np.where(is_to_right, obstacles_expanded[:, 0, :], robot_sklt_projections[:, :, 0])
        robot_sklt_projections[:, :, 0] = np.where(is_to_left, obstacles_expanded[:, 1, :], robot_sklt_projections[:, :, 0])

        # if is_infront y_projection = y_max; if is_behind y_projection = y_min
        robot_sklt_projections[:, :, 1] = np.where(is_infront, obstacles_expanded[:, 2, :], robot_sklt_projections[:, :, 1])
        robot_sklt_projections[:, :, 1] = np.where(is_behind, obstacles_expanded[:, 3, :], robot_sklt_projections[:, :, 1])

        # if is_above z_projection = z_max; if is_below z_projection = z_min
        robot_sklt_projections[:, :, 2] = np.where(is_above, obstacles_expanded[:, 4, :], robot_sklt_projections[:, :, 2])
        robot_sklt_projections[:, :, 2] = np.where(is_below, obstacles_expanded[:, 5, :], robot_sklt_projections[:, :, 2])

        # take the squared difference between projection and origin
        distances_proj_origin = np.square(robot_sklt_projections - robot_sklt)

        # set the differences for the projections on the table for the skeleton points that should ignore collision
        # with the table to infinity; this will make sure that these are not selected as the minimal distances
        # note that this assumes the table to be the obstacle at index 0
        distances_proj_origin[0, self.sklt_indx_ignore_table, :] = np.inf

        # finish the computation of the distances between projection and origin by summing and taking the root
        distances_proj_origin = np.sqrt(distances_proj_origin.sum(axis=2))

        # retrieve the closest obstacle cuboid
        min_idx = distances_proj_origin.min(axis=1).argmin()
        self.closest_obstacle_cuboid = obstacle_cuboids[min_idx, :].astype(np.float32)

        # set the shortest distance to obstacles
        self.min_distance_to_obstacles = distances_proj_origin.min()

        # display closest obstacle cuboid
        if self.debug["closest_obstacle_cuboid"]:
            pyb.removeAllUserDebugItems()
            # get edge values
            x_max, x_min, y_max, y_min, z_max, z_min = self.closest_obstacle_cuboid[:6]

            # function wrap
            def draw_line(lineFrom, lineTo):
                pyb.addUserDebugLine(lineFrom, lineTo, lineWidth=3, lineColorRGB=[0, 0, 255])

            # draw from corner to corner
            draw_line([x_min, y_max, z_max], [x_min, y_max, z_max])
            draw_line([x_min, y_max, z_max], [x_min, y_min, z_max])
            draw_line([x_min, y_max, z_max], [x_min, y_max, z_min])

            draw_line([x_min, y_min, z_max], [x_min, y_max, z_max])
            draw_line([x_min, y_min, z_max], [x_min, y_min, z_min])
            draw_line([x_min, y_min, z_max], [x_max, y_min, z_min])

            draw_line([x_max, y_max, z_min], [x_min, y_max, z_min])
            draw_line([x_max, y_max, z_min], [x_max, y_min, z_min])
            draw_line([x_max, y_max, z_min], [x_max, y_max, z_max])

            draw_line([x_min, y_min, z_min], [x_max, y_min, z_min])
            draw_line([x_min, y_min, z_min], [x_min, y_max, z_min])
            draw_line([x_min, y_min, z_min], [x_min, y_min, z_max])
