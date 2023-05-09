import pybullet as pyb
from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from time import process_time
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from typing import List

__all__ = [
    "ObstacleSensor"
]

class ObstacleSensor(Sensor):
    """
    This sensor reports the relative position of obstacles in the vicinity.
    To make the measurements consistent, it will spawn a small invisible and non-colliding sphere at a probe location, which it will then use to measure the distances.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, sim_steps_per_env_step: int, robot: Robot, num_obstacles: int, max_distance: float, reference_link_ids: List[str], sphere_coordinates: bool=False):
        
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, sim_steps_per_env_step)

        # set associated robot
        self.robot = robot

        # num of obstacles reported in the output
        self.num_obstacles = num_obstacles
        # maximum distance which the sensor will consider for obstacles
        self.max_distance = max_distance
        # default observation
        # 0 0 0 vector, gets used when there are not enough obstacles in the env currently to fill out the observation
        self.default_observation = np.array([[0, 0, 0] for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        # list of link ids for which the sensor will work
        self.reference_link_ids = reference_link_ids

        # set output data field name
        self.output_name = "nearest_" + str(self.num_obstacles) + "_obstacles_" + self.robot.name
        self.output_name_time = "obstacle_sensor_cpu_time_" + self.robot.name

        # probe object
        self.default_position = np.array([0, 0, -10])
        self.probe = pyb_u.create_sphere(self.default_position, 0, 0.001, color = [0.5, 0.5, 0.5, 0.0001], collision=True)  

        # init data storage
        self.output_vector = np.tile(self.default_observation, (len(self.reference_link_ids), 1))
        self.data_raw = [None for _ in self.reference_link_ids]

        # attributes for outside access
        self.min_dist = np.inf

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / (np.ones(3 * self.num_obstacles) * self.max_distance * 2)
        self.normalizing_constant_b = np.ones(3 * self.num_obstacles) - np.multiply(self.normalizing_constant_a, np.ones(3 * self.num_obstacles) * self.max_distance)
        self.normalizing_constant_a_spherical = 2 / np.tile([max_distance, 4 * np.pi, 4 * np.pi], self.num_obstacles)
        self.normalizing_constant_b_spherical = np.ones(3 * self.num_obstacles) - np.multiply(self.normalizing_constant_a_spherical, np.tile([max_distance, 2 * np.pi, 2 * np.pi], self.num_obstacles))
        # bool for reporting output in spherical coordinates
        self.sphere_coordinates = sphere_coordinates

    def update(self, step) -> dict:
        self.cpu_epoch = process_time()
        if step % self.update_steps == 0:
            self.min_dist = np.inf
            for idx, link in enumerate(self.reference_link_ids):
                link_position, _, _, _ = pyb_u.get_link_state(self.robot.object_id, link)
                pyb_u.set_base_pos_and_ori(self.probe, link_position, np.array([0, 0, 0, 1]))

                self.output_vector[idx] = self.default_observation
                self.data_raw[idx] = self._run_obstacle_detection()
                new_data = self._process(self.data_raw[idx])
                self.output_vector[idx][:len(new_data)] = new_data
                if self.sphere_coordinates:
                    self.min_dist = min(self.min_dist, self.output_vector[idx][0])
                else:
                    self.min_dist = min(self.min_dist, np.linalg.norm(self.output_vector[idx][:3]))
            pyb_u.set_base_pos_and_ori(self.probe, self.default_position, np.array([0, 0, 0, 1]))
        self.cpu_time = process_time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = process_time()
        self.output_vector = np.tile(self.default_observation, (len(self.reference_link_ids), 1))

        self.min_dist = np.inf
        for idx, link in enumerate(self.reference_link_ids):
            link_position, _, _, _ = pyb_u.get_link_state(self.robot.object_id, link)
            pyb_u.set_base_pos_and_ori(self.probe, link_position, np.array([0, 0, 0, 1]))

            self.output_vector[idx] = self.default_observation
            self.data_raw[idx] = self._run_obstacle_detection()
            new_data = self._process(self.data_raw[idx])
            self.output_vector[idx][:len(new_data)] = new_data
            if self.sphere_coordinates:
                self.min_dist = min(self.min_dist, self.output_vector[idx][0])
            else:
                self.min_dist = min(self.min_dist, np.linalg.norm(self.output_vector[idx][:3]))
        pyb_u.set_base_pos_and_ori(self.probe, self.default_position, np.array([0, 0, 0, 1]))
        self.cpu_time = process_time() - self.cpu_epoch
        self.aux_visual_objects = []

    def get_observation(self) -> dict:
        if self.normalize:
            return self._normalize()
        else:
            ret_dict = dict()
            for idx, link in enumerate(self.reference_link_ids):
                ret_dict[self.output_name + "_" + link] = self.output_vector[idx]
            return ret_dict

    def _normalize(self) -> dict:
        ret_dict = dict()
        for idx, link in enumerate(self.reference_link_ids):
            if self.sphere_coordinates:
                ret_dict[self.output_name + "_" + link] = np.multiply(self.normalizing_constant_a_spherical, self.output_vector[idx]) + self.normalizing_constant_b_spherical
            else:
                ret_dict[self.output_name + "_" + link] = np.multiply(self.normalizing_constant_a, self.output_vector[idx]) + self.normalizing_constant_b
        return ret_dict

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret_dict = dict()
            if self.normalize:
                for link in self.reference_link_ids:
                    ret_dict[self.output_name + "_" + link] = Box(low=-1, high=1, shape=(3 * self.num_obstacles,), dtype=np.float32)
            else:
                for link in self.reference_link_ids:
                    ret_dict[self.output_name + "_" + link] = Box(low=-self.max_distance, high=self.max_distance, shape=(3 * self.num_obstacles,), dtype=np.float32)
            return ret_dict
        else:
            return {}

    def _run_obstacle_detection(self):

        res = []
        # get nearest robots
        for object_id in pyb_u.pybullet_object_ids:
            if object_id != self.probe and object_id != self.robot.object_id:
                closestPoints = pyb.getClosestPoints(pyb_u.to_pb(self.probe), pyb_u.to_pb(object_id), self.max_distance)
                if not closestPoints:
                    continue
                min_val = min(closestPoints, key=lambda x: x[8])  # index 8 is the distance in the object returned by pybullet
                res.append(np.hstack([np.array(min_val[5]), np.array(min_val[6]), min_val[8]]))  # start, end, distance
        # sort
        res.sort(key=lambda x: x[6])
        # extract n closest ones
        smallest_n = res[:self.num_obstacles]

        return np.array(smallest_n)

    def _process(self, data_raw):
        data_processed = []
        for i in range(len(data_raw)):
            vector = data_raw[i][3:6] -  data_raw[i][0:3]
            if self.sphere_coordinates:
                r = np.linalg.norm(vector)
                theta = np.arccos(vector[2]/r)
                phi = np.arctan2(vector[1], vector[0])
                vector = np.array([r, theta, phi])
            data_processed.append(vector)
        return np.array(data_processed).flatten()

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict[self.output_name] = self.output_vector
        logging_dict[self.output_name_time] = self.cpu_time

        return logging_dict

    def build_visual_aux(self):

        for idx, _ in enumerate(self.reference_link_ids):
            line_starts = [self.data_raw[idx][i][0:3] for i in range(len(self.data_raw[idx]))]
            line_ends = [self.data_raw[idx][i][3:6] for i in range(len(self.data_raw[idx]))]
            colors = [[0, 0, 1] for _ in range(len(self.data_raw[idx]))]

            self.aux_lines += pyb_u.draw_lines(line_starts, line_ends, colors)
