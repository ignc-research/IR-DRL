import gym.spaces

from goal.goal import Goal
import numpy as np
from gym.spaces import Box
import pybullet as pyb
from numpy import newaxis as na
import time


class PositionCollisionTrajectoryPCR(Goal):
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

        # get trajectories
        import json
        with open(goal_config["trajectories"], "r") as f:
            self.trajectory_data = json.load(f)
        self.trajectories = self.trajectory_data["trajectories"]
        self.trajectory_targets = np.asarray(self.trajectory_data["targets"])

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

        # set the distance thresholds and the increments for changing them
        self.distance_threshold = goal_config["dist_threshold_start"] if self.train else goal_config[
            "dist_threshold_end"]
        self.distance_threshold_start = goal_config["dist_threshold_start"]
        self.distance_threshold_end = goal_config["dist_threshold_end"]
        self.distance_threshold_increment_start = goal_config["dist_threshold_increment_start"]
        self.distance_threshold_increment_end = goal_config["dist_threshold_increment_end"]

        # workspace bounderies
        self.boundaries = goal_config["boundaries"]
        self.boundaries_min = np.array([self.boundaries[0], self.boundaries[2], self.boundaries[4]])
        self.boundaries_max = np.array([self.boundaries[1], self.boundaries[3], self.boundaries[5]])
        self.boundaries_range = self.boundaries_max - self.boundaries_min
        # performance metric name
        self.metric_name = "distance_threshold"

        # set indices of the robot skeleton points that should ignore the table
        self.sklt_indx_ignore_table = [0]

        # set indices of the robot skeleton points that should consider the table
        self.sklt_indx_consider_table = [1, 2]

        # stuff for debugging
        self.debug = goal_config["debug"]

        # we initialize the encoded obstacle with a made up obstacle that is far from the robots reach
        self.obstacle_encoded = self.encode_cuboid_pcr(
                np.array([-1.99, -2, -1.99, -2, 2.99, 3, 0.01, 0.01, 0.01, -1.995, -1.995, -2.995]))
    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            ret["closest_projection_spherical"] = Box(low=-1, high=1, shape=(3, 3), dtype=np.float32)
            ret["qt"] = Box(low=-1, high=1, shape=(6,1), dtype=np.float32)
            ret["f"] = gym.spaces.MultiBinary(1)
            ret["qt_qr_delta"] = Box(low=-1, high=1, shape=(6,), dtype=np.float32)
            return ret
        else:
            return {}

    def on_env_reset(self, success_rate, episode):
        t = time.time()
        self.timeout = False
        self.prev_action = np.array([0, 0, 0, 0, 0])
        self.is_success = False
        self.success = False
        self.collided = False
        self.done = False
        self.ep_reward = 0
        # get trajectory for target
        self.target = self.robot.world.position_targets[self.robot.id]
        traj_indx = np.argwhere(np.all(self.target == self.trajectory_targets, axis=1))
        traj_indx = traj_indx[np.random.randint(0, len(traj_indx))]
        self.trajectory = self.trajectories[traj_indx.item()]
        self.trajectory = np.asarray(self.trajectory, dtype=np.float32)
        self.waypoint = self.trajectory[-1]
        # set observations
        self._set_observation(0)

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
        return self.metric_name, 0, False, True

    def normalize_spherical_coordinates(self, x):
        # normalize r
        x[:, 0] = np.clip(x[:, 0], -1.55, 1.55) / 1.55
        # normalize teta
        x[:, 1] = x[:, 1] / np.pi
        # normalize phi
        x[:, 2] = x[:, 2] / (np.pi / 2)
        return x
    def get_observation(self) -> dict:
        if self.normalize_observations:
            closest_projection_spherical = self.normalize_spherical_coordinates(self.closest_projection_spherical)
            qt = self.qt / np.pi
            qt_qr_delta = self.qt_qr_delta / (2*np.pi)
            self.f = np.asarray([self.f], dtype=int)
            return {"closest_projection_spherical": closest_projection_spherical,
                    "qt": qt[:, na],
                    "f": self.f,
                    "qt_qr_delta": qt_qr_delta}

    def _set_observation(self, step):
        self.qt = self.trajectory[np.clip(step, 1, len(self.trajectory) - 1)]
        self.f = True if np.all(self.qt == self.waypoint) else False
        self.robot.set_trajectory_point(self.qt)
        self.qt_qr_delta = self.qt - self.robot.joints_sensor.joints_angles
        self._set_min_distance_to_obstacle_and_closest_cuboid()

    def reward(self, step, action):
        t = time.time()

        # set observations
        self._set_observation(step + 1)

        if not self.collided:
            self.collided = self.robot.world.collision

        # binary signal for when robot is too close to obstacle
        d_min = 0.225
        if self.min_distance_to_obstacles < d_min:
            r_dist = 1
        else:
            r_dist = 0

        # reward for staying close to the predefined trajectory
        qt_qr_delta_sum = np.absolute(self.qt_qr_delta[:5]).sum()
        threshhold_t = 10
        r_delta = - (1/threshhold_t) * qt_qr_delta_sum + 1
        r_delta = np.clip(r_delta, 0, 1)

        # reward for action
        r_a = np.mean(np.square(action))
        threshhold_a_dot = 0.2
        r_a_dot = np.sum(np.where(np.absolute(action - self.prev_action) > threshhold_a_dot, 1, 0))

        # reward for reaching a waypoint
        self.is_success = False
        # can only succeed if not collided
        if self.collided:
            self.done = True
            r_collision = 1
            r_waypoint = 0
        else:
            r_collision = 0
            # if success
            if np.all(np.absolute(self.robot.joints_sensor.joints_angles[0:5] - self.qt[0:5]) < self.distance_threshold):
                r_waypoint = 1
                self.is_success = True
                self.done = True
            else:
                r_waypoint = 0

        w_dist = -0.002
        w_delta = 0.001
        w_a = -0.0015
        w_a_dot = -0.0004
        w_waypoint = 0.05
        w_collision = -0.1

        reward = w_dist * r_dist + w_delta * r_delta + w_a * r_a + w_a_dot * r_a_dot + w_waypoint * r_waypoint + w_collision * r_collision
        if step > self.max_steps:
            self.done = True

        self.reward_value = reward
        self.ep_reward += reward
        self.cpu_epoch = time.time() - t

        self.prev_action = action
        return self.reward_value, self.is_success, self.done, self.timeout, False

    def build_visual_aux(self):
        pass

    def get_data_for_logging(self) -> dict:
        logging_dict = dict()
        logging_dict["reward_" + self.robot.name] = self.reward_value
        logging_dict["min_distance_to_obstacles"] = self.min_distance_to_obstacles
        logging_dict["ep_reward"] = self.ep_reward
        logging_dict["goal_cpu_time"] = self.cpu_epoch
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold
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

        # closest projection in spherical coordinates with respect to shoulder, elbow and ee
        delta = robot_sklt - self.closest_projection

        self.closest_projection_spherical = np.empty((3, 3), dtype=np.float32)
        # r
        self.closest_projection_spherical[:, 0] = np.sqrt(np.sum(np.square(delta), axis=1))
        # teta
        self.closest_projection_spherical[:, 1] = np.arccos(delta[:, 2] / (self.closest_projection_spherical[:, 0] + 0.0000001))
        # phi
        self.closest_projection_spherical[:, 2] = np.arctan(delta[:, 1] / (delta[:, 0] + 0.0000001))

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

