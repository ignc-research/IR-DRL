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

        try:
            # if the full skeleton should be added to the observation
            self.add_full_skeleton_to_obs = goal_config["add_full_skeleton_to_obs"]
        except KeyError:
            self.add_full_skeleton_to_obs = False

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

        # workspace bounderies
        self.boundaries = goal_config["boundaries"]
        self.boundaries_min = np.array([self.boundaries[0], self.boundaries[2], self.boundaries[4]])
        self.boundaries_max = np.array([self.boundaries[1], self.boundaries[3], self.boundaries[5]])
        self.boundaries_range = self.boundaries_max - self.boundaries_min
        # performance metric name
        self.metric_name = "distance_threshold"

        # set indices of the robot skeleton points that should ignore the table
        self.sklt_indx_ignore_table = [0, 1, 2, 3, 10, 11, 12]

        # set indices of the robot skeleton points that should consider the table
        self.sklt_indx_consider_table = [4, 5, 6, 7, 8, 9, 13, 14, 15]

        # stuff for debugging
        self.debug = goal_config["debug"]

        # we initialize the encoded obstacle with a made up obstacle that is far from the robots reach
        self.obstacle_encoded = self.encode_cuboid_pcr(
                np.array([-1.99, -2, -1.99, -2, 2.99, 3, 0.01, 0.01, 0.01, -1.995, -1.995, -2.995]))
    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            if not self.add_full_skeleton_to_obs:
                ret = dict()
                ret["target_position"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
                ret["end_effector_position"] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                ret["ee_target_delta"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
                ret["closest_robot_sklt_point"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
                ret["closest_projection"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
                ret["sklt_projection_delta"] = Box(low=-1, high=1, shape=(3, ), dtype=np.float32)
            else:
                ret = dict()
                ret["target_position"] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                ret["ee_target_delta"] = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                ret["robot_sklt"] = Box(low=-1, high=1, shape=(16, 3), dtype=np.float32)
                ret["robot_sklt_projections"] = Box(low=-1, high=1, shape=(16, 3), dtype=np.float32)
                ret["sklt_projection_delta"] = Box(low=-1, high=1, shape=(16, 3), dtype=np.float32)
            return ret
        else:
            return {}

    def on_env_reset(self, success_rate, episode):
        t = time.time()
        self.timeout = False
        self.is_success = False
        self.success = False
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
            if success_rate > 0.8 and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
            elif success_rate < 0.8 and self.distance_threshold < self.distance_threshold_start:
                self.distance_threshold += increment / 25  # upwards movement should be slower
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end

        self.cpu_epoch = time.time() - t
        return self.metric_name, self.distance_threshold, True, True

    def normalize_coordinates(self, points):
        points = np.clip(points, a_min=self.boundaries_min, a_max=self.boundaries_max)
        points = 2 * (points - self.boundaries_min) / (self.boundaries_max - self.boundaries_min) - 1
        return points


    def get_observation(self) -> dict:
        if self.normalize_observations:
            if not self.add_full_skeleton_to_obs:
                target_pos = self.normalize_coordinates(self.target)
                end_effector_position = self.normalize_coordinates(self.position)
                ee_target_delta = (target_pos - end_effector_position) / 2
                closest_robot_sklt_point = self.normalize_coordinates(self.closest_robot_skeleton_point)
                closest_projection = self.normalize_coordinates(self.closest_projection)
                sklt_projection_delta = (closest_projection - closest_robot_sklt_point) / 2

                return {"target_position": target_pos,
                        "end_effector_position": end_effector_position,
                        "ee_target_delta": ee_target_delta,
                        "closest_robot_sklt_point": closest_robot_sklt_point,
                        "closest_projection": closest_projection,
                        "sklt_projection_delta": sklt_projection_delta
                        }

            else:
                target_pos = self.normalize_coordinates(self.target)
                end_effector_position = self.normalize_coordinates(self.position)
                ee_target_delta = (target_pos - end_effector_position) / 2
                robot_sklt = self.normalize_coordinates(self.robot_skeleton_sensor.robot_skeleton)
                robot_sklt_projections = self.normalize_coordinates(self.closest_projections)
                sklt_projection_delta = (robot_sklt_projections - robot_sklt) / 2
                return {"target_position": target_pos,
                        "ee_target_delta": ee_target_delta,
                        "robot_sklt": robot_sklt,
                        "robot_sklt_projections": robot_sklt_projections,
                        "sklt_projection_delta": sklt_projection_delta
                        }

    def _set_observation(self):
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.target = self.robot.world.position_targets[self.robot.id].astype(np.float32)
        dif = self.target - self.position
        self.ee_target_delta = self.position - self.target
        self.distance = np.array([np.linalg.norm(dif)])
        self._set_min_distance_to_obstacle_and_closest_cuboid()

    def reward(self, step, action):
        t = time.time()

        reward = 0

        if not self.collided:
            self.collided = self.robot.world.collision

        # set parameters
        lambda_1 = 1000
        lambda_2 = 100
        lambda_3 = 60
        # reward_out = 0.01
        dirac = 0.1
        k = 8
        d_ref = 0.33

        # set observations
        self._set_observation()
        # set motion change
        a = action  # note that the action is normalized

        # calculating Huber loss for distance of end effector to target
        if self.distance > 1.55: self.distance = np.array([1.55])
        if self.distance < dirac:
            R_E_T = 1 / 2 * (self.distance ** 2)
        else:
            R_E_T = dirac * (self.distance - 1 / 2 * dirac)
        R_E_T = -R_E_T

        min_distance_to_obstacles = self.min_distance_to_obstacles
        # calculating the distance between robot and obstacle
        R_R_O = -(d_ref / (min_distance_to_obstacles + d_ref)) ** k

        # calculate motion size
        R_A = - np.sum(np.square(a))

        # success
        self.is_success = False
        if self.collided:
            self.done = True
            reward += -100
        elif self.distance[0] < self.distance_threshold:
            self.done = True
            self.is_success = True
            reward += 100
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += -50
        else:
            if self.normalize_rewards:
                reward = (lambda_1 * (R_E_T / 0.15) + lambda_2 * (R_R_O) + lambda_3 * (R_A / 6)) / lambda_1
                #print("Distance reward:", self.distance, (R_E_T / 0.15))
                #print("Distance to obstacle reward:", self.min_distance_to_obstacles, lambda_2 * (R_R_O) / lambda_1)
                #print("Motion size reward:", a, lambda_3 * (R_A / 6) / lambda_1)
            else:
                # calculate reward
                reward = lambda_1 * R_E_T + lambda_2 * R_R_O + lambda_3 * R_A

        self.reward_value = reward
        self.ep_reward += reward
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
        logging_dict["distance_" + self.robot.name] = self.distance[0]
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold
        logging_dict["ep_reward"] = self.ep_reward
        logging_dict["goal_cpu_time"] = self.cpu_epoch
        return logging_dict

    def _set_min_distance_to_obstacle_and_closest_cuboid(self):
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
        robot_sklt_projections[:, :, 0] = np.where(is_to_right, obstacles_expanded[:, 0, :],
                                                   robot_sklt_projections[:, :, 0])
        robot_sklt_projections[:, :, 0] = np.where(is_to_left, obstacles_expanded[:, 1, :],
                                                   robot_sklt_projections[:, :, 0])

        # if is_infront y_projection = y_max; if is_behind y_projection = y_min
        robot_sklt_projections[:, :, 1] = np.where(is_infront, obstacles_expanded[:, 2, :],
                                                   robot_sklt_projections[:, :, 1])
        robot_sklt_projections[:, :, 1] = np.where(is_behind, obstacles_expanded[:, 3, :],
                                                   robot_sklt_projections[:, :, 1])

        # if is_above z_projection = z_max; if is_below z_projection = z_min
        robot_sklt_projections[:, :, 2] = np.where(is_above, obstacles_expanded[:, 4, :],
                                                   robot_sklt_projections[:, :, 2])
        robot_sklt_projections[:, :, 2] = np.where(is_below, obstacles_expanded[:, 5, :],
                                                   robot_sklt_projections[:, :, 2])

        # take the squared difference between projection and origin
        distances_proj_origin = np.square(robot_sklt_projections - robot_sklt)

        # set the differences for the projections on the table for the skeleton points that should ignore collision
        # with the table to infinity; this will make sure that these are not selected as the minimal distances
        # note that this assumes the table to be the obstacle at index 0
        distances_proj_origin[0, self.sklt_indx_ignore_table, :] = np.inf

        # finish the computation of the distances between projection and origin by summing and taking the root
        distances_proj_origin = np.sqrt(distances_proj_origin.sum(axis=2))

        # index of the closest obstacle cuboid
        min_idx_cuboid = distances_proj_origin.min(axis=1).argmin()

        # index of the robot skeleton point that is the closest to the cuboid
        min_idk_sklt = distances_proj_origin[min_idx_cuboid, :].argmin()

        # get closest projection
        self.closest_projection = robot_sklt_projections[min_idx_cuboid, min_idk_sklt, :]
        self.closest_projections = robot_sklt_projections[min_idx_cuboid, :, :]

        # closest robot skeleton point
        self.closest_robot_skeleton_point = robot_sklt[min_idk_sklt, :]

        # closest cuboid for debugging
        self.closest_obstacle_cuboid = obstacle_cuboids[min_idx_cuboid, :].astype(np.float32)
        # set the shortest distance to obstacles
        self.min_distance_to_obstacles = distances_proj_origin.min()

        # if len(obstacle_cuboids) != 1:
        #     # encode obstacle cuboid
        #     self.obstacle_encoded = self.encode_cuboid_pcr(cuboid=obstacle_cuboids[1])

        # colors = np.repeat(np.array([0, 0, 255])[na, :], len(self.obstacle_encoded), axis=0)
        # pyb.addUserDebugPoints(np.asarray(self.obstacle_encoded), colors, pointSize=2)
        # time.sleep(352343)

        # display closest obstacle cuboid
        if self.debug["closest_obstacle_cuboid"]:
            pyb.removeAllUserDebugItems()
            # get edge values
            x_max, x_min, y_max, y_min, z_max, z_min = self.closest_obstacle_cuboid[:6]

            # function wrap
            def draw_line(lineFrom, lineTo):
                pyb.addUserDebugLine(lineFrom,
                                     lineTo,
                                     lineWidth=3,
                                     lineColorRGB=[0, 0, 255],
                                     lifeTime=10)

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

            # draw additional line starting from center point
            draw_line(self.closest_obstacle_cuboid[-3:],
                      self.closest_obstacle_cuboid[-3:] + np.array([0, 0, 0.3]))

    @staticmethod
    def encode_cuboid_pcr(cuboid, points_per_plane=16):
        # has to have an even square root
        assert np.sqrt(points_per_plane).is_integer()

        x_max, x_min, y_max, y_min, z_max, z_min, length, depth, height, x_center, y_center, z_center = cuboid
        length = x_max - x_min
        depth = y_max - y_min
        height = z_max - z_min

        def create_plane(min1, max1, axis1, min2, max2, axis2, const_val, axis_const, n_points=points_per_plane):
            n = np.sqrt(points_per_plane).astype(int)
            plane_points = np.empty((n_points, 3))
            A, B = np.mgrid[
                   min1:max1:n * 1j,
                   min2:max2:n * 1j
                   ]
            A = A.reshape((n_points, ))
            B = B.reshape((n_points, ))

            plane_points[:, axis1] = A
            plane_points[:, axis2] = B
            plane_points[:, axis_const] = const_val

            return plane_points

        front_plane = create_plane(x_min, x_max, 0, z_min, z_max, 2, y_min, 1)
        back_plane = front_plane + np.array([0, depth, 0])

        right_plane = create_plane(y_min, y_max, 1, z_min, z_max, 2, x_max, 0)
        left_plane = right_plane - np.array([length, 0, 0])

        top_plane = create_plane(x_min, x_max, 0, y_min, y_max, 1, z_max, 2)
        bottom_plane = top_plane - np.array([0, 0, height])

        planes = [front_plane, back_plane, right_plane, left_plane, top_plane, bottom_plane]
        cuboid_pcr = np.concatenate(planes, axis=0)

        return cuboid_pcr

