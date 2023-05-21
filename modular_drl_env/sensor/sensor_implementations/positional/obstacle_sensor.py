import pybullet as pyb
from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from time import process_time
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from typing import List
from modular_drl_env.world.obstacles.ground_plate import GroundPlate

__all__ = [
    "ObstacleSensor"
]

class ObstacleSensor(Sensor):
    """
    This sensor reports the relative position of obstacles in the vicinity.
    To make the measurements consistent, it will spawn a small invisible and non-colliding sphere at a probe location, which it will then use to measure the distances.
    """

    def __init__(self, normalize: bool, 
                 add_to_observation_space: bool, 
                 add_to_logging: bool, 
                 sim_step: float, 
                 update_steps: int, 
                 sim_steps_per_env_step: int, 
                 robot: Robot, 
                 num_obstacles: int, 
                 max_distance: float, 
                 reference_link_ids: List[str], 
                 sphere_coordinates: bool=False, 
                 report_velocities: bool=False):
        
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, sim_steps_per_env_step)

        # set associated robot
        self.robot = robot

        # num of obstacles reported in the output
        self.num_obstacles = num_obstacles
        # maximum distance which the sensor will consider for obstacles
        self.max_distance = max_distance
        # default observation
        # 0 0 0 vector, gets used when there are not enough obstacles in the env currently to fill out the observation
        self.default_observation = np.array([[2*max_distance, 2*max_distance, 2*max_distance] for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        self.default_observation_vels = np.array([[0, 0, 0] for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
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
        self.output_vector_vels = np.tile(self.default_observation, (len(self.reference_link_ids), 1))
        self.data_raw = [None for _ in self.reference_link_ids]

        # attributes for outside access
        self.min_dist = np.inf

        # report obstacle velocities
        self.report_velocities = report_velocities

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
                self.output_vector_vels[idx] = self.default_observation_vels
                self.data_raw[idx] = self._run_obstacle_detection(link)
                new_data, new_vels = self._process(self.data_raw[idx])
                self.output_vector[idx][:len(new_data)] = new_data
                if self.report_velocities:
                    self.output_vector_vels[idx][:len(new_vels)] = new_vels
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
            self.output_vector_vels[idx] = self.default_observation_vels
            self.data_raw[idx] = self._run_obstacle_detection(link)
            new_data, new_vels = self._process(self.data_raw[idx])
            self.output_vector[idx][:len(new_data)] = new_data
            if self.report_velocities:
                self.output_vector_vels[idx][:len(new_vels)] = new_vels
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
                if self.report_velocities:
                    ret_dict[self.output_name + "_" + link + "_velocities"] = self.output_vector_vels[idx]
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
                    if self.report_velocities:
                        ret_dict[self.output_name + "_" + link + "_velocities"] = Box(low=-1, high=1, shape=(3 * self.num_obstacles,), dtype=np.float32)
            else:
                for link in self.reference_link_ids:
                    ret_dict[self.output_name + "_" + link] = Box(low=-self.max_distance, high=self.max_distance, shape=(3 * self.num_obstacles,), dtype=np.float32)
                    if self.report_velocities:
                        ret_dict[self.output_name + "_" + link + "_velocities"] = Box(low=-100, high=100, shape=(3 * self.num_obstacles,), dtype=np.float32)
            return ret_dict
        else:
            return {}

    def _run_obstacle_detection(self, link):

        res = []
        # get nearest robots
        for object in self.robot.world.active_objects:
            closestPoints = pyb.getClosestPoints(pyb_u.to_pb(self.probe), pyb_u.to_pb(object.object_id), self.max_distance)
            if not closestPoints:
                continue
            min_val = min(closestPoints, key=lambda x: x[8])  # index 8 is the distance in the object returned by pybullet
            if self.report_velocities:
                _, _, own_velo, _ = pyb_u.get_link_state(self.robot.object_id, link)
                rel_velo = object.velocity - own_velo  # relative velocity of obstalce w.r.t. to link
                res.append(np.hstack([np.array(min_val[5]), np.array(min_val[6]), min_val[8], rel_velo]))  # start, end, distance, velocity
            else:
                res.append(np.hstack([np.array(min_val[5]), np.array(min_val[6]), min_val[8]]))

        # sort
        res.sort(key=lambda x: x[6])
        # extract n closest ones
        smallest_n = res[:self.num_obstacles]

        return np.array(smallest_n)

    def _process(self, data_raw):
        data_processed = []
        vels_processed = []
        for i in range(len(data_raw)):
            vector = data_raw[i][3:6] -  data_raw[i][0:3]
            if self.sphere_coordinates:
                r = np.linalg.norm(vector)
                theta = np.arccos(vector[2]/r)
                phi = np.arctan2(vector[1], vector[0])
                vector = np.array([r, theta, phi])
            data_processed.append(vector)
            if self.report_velocities:
                vels_processed.append(data_raw[i][7:10])
        return np.array(data_processed).flatten(), np.array(vels_processed).flatten()

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

class ObstacleAbsoluteSensor(Sensor):

    def __init__(self, 
                 normalize: bool, 
                 add_to_observation_space: bool, 
                 add_to_logging: bool, 
                 sim_step: float, 
                 update_steps: int, 
                 sim_steps_per_env_step: int, 
                 robot: Robot, 
                 num_obstacles: int, 
                 max_distance: float, 
                 report_velocities: bool=False,
                 ignore_ground: bool=True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, sim_steps_per_env_step)

        self.robot = robot

        self.num_obstacles = num_obstacles
        self.max_distance = max_distance
        self.report_velocities = report_velocities

        self.def_obs_pos = np.array([np.array([0, 0, -10]) for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        self.def_obs_vel = np.array([np.array([0, 0, 0,]) for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        self.def_obs_dist = np.array([self.max_distance for _ in range(self.num_obstacles)], dtype=np.float32).flatten()

        max_vec = np.array([[self.robot.world.x_max, self.robot.world.y_max, self.robot.world.z_max] for _ in range(self.num_obstacles)]).flatten()
        min_vec = np.array([[self.robot.world.x_min, self.robot.world.y_min, self.robot.world.z_min] for _ in range(self.num_obstacles)]).flatten()
        range_vec = max_vec - min_vec
        self.normalizing_constant_a_pos = 2 / range_vec
        self.normalizing_constant_b_pos = np.ones(3 * self.num_obstacles) - np.multiply(self.normalizing_constant_a_pos, range_vec)
        self.normalizing_constant_a_vel = 2 / (np.ones(3 * self.num_obstacles) * 50)
        self.normalizing_constant_b_vel = np.ones(3 * self.num_obstacles) - np.multiply(self.normalizing_constant_a_pos, np.ones(3 * self.num_obstacles) * 25)

        self.positions = np.array([np.zeros(3 * self.num_obstacles) for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        self.velocities = np.array([np.zeros(3 * self.num_obstacles) for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        self.distances = np.zeros(self.num_obstacles, dtype=np.float32)

        self.min_dist = np.inf

        self.ignore_ground = ignore_ground

    def get_observation_space_element(self) -> dict:
        ret = dict()
        if self.normalize:
            ret["obstacle_positions"] = Box(low=-1, high=1, shape=(3 * self.num_obstacles,), dtype=np.float32)
            ret["obstacle_velocities"] = Box(low=-1, high=1, shape=(3 * self.num_obstacles,), dtype=np.float32)
            ret["obstacle_distances"] = Box(low=0, high=1, shape=(self.num_obstacles,), dtype=np.float32)
        else:
            low = np.array([[self.robot.world.x_min, self.robot.world.y_min, self.robot.world.z_min] for _ in range(self.num_obstacles)]).flatten()
            high = np.array([[self.robot.world.x_max, self.robot.world.y_max, self.robot.world.z_max] for _ in range(self.num_obstacles)]).flatten()
            ret["obstacle_positions"] = Box(low=low, high=high, shape=(3 * self.num_obstacles,), dtype=np.float32)
            ret["obstacle_velocities"] = Box(low=-25, high=25, shape=(3 * self.num_obstacles,), dtype=np.float32)
            ret["obstacle_distances"] = Box(low=0, high=self.max_distance, shape=(self.num_obstacles,), dtype=np.float32)
        return ret

    def get_observation(self) -> dict:
        ret = dict()
        if self.normalize:
            ret["obstacle_positions"] = np.multiply(self.normalizing_constant_a_pos, self.positions) + self.normalizing_constant_b_pos
            ret["obstacle_velocities"] = np.multiply(self.normalizing_constant_a_vel, self.velocities) + self.normalizing_constant_b_vel
            ret["obstacle_distances"] = self.distances / self.max_distance
        else:
            ret["obstacle_positions"] = self.positions
            ret["obstacle_velocities"] = self.velocities
            ret["obstacle_distances"] = self.distances
        return ret
    
    def update(self, step) -> dict:
        cpu_epoch = process_time()
        if step % self.update_steps == 0:
            self._get_data()
        self.cpu_time = process_time() - cpu_epoch
        return self.get_observation()

    def reset(self):
        cpu_epoch = process_time()
        self._get_data()
        self.cpu_time = process_time() - cpu_epoch
    
    def _get_data(self):
        # check the distances of all active obstacles
        candidates = []
        for obstacle in self.robot.world.active_objects:
            if self.ignore_ground and type(obstacle) == GroundPlate:
                continue
            pyb_data = pyb.getClosestPoints(pyb_u.to_pb(self.robot.object_id), pyb_u.to_pb(obstacle.object_id), self.max_distance)
            closest_element = min(pyb_data, key=lambda x: x[8])
            closestDist = closest_element[8]
            candidates.append((closestDist, obstacle))
        # set output to default first, this is useful for when we have less obstacles in radius than the size of our output
        self.positions = self.def_obs_pos
        self.velocities = self.def_obs_vel
        self.distances = self.def_obs_dist
        # now take the closest ones and get their data
        candidates.sort(key=lambda x: x[0])
        if len(candidates) != 0:
            self.min_dist = candidates[0][0]
        for idx, obst in enumerate(candidates[:self.num_obstacles]):
            self.positions[idx * 3: (idx + 1) * 3] = obst[1].position
            self.velocities[idx * 3: (idx + 1) * 3] = obst[1].velocity
            self.distances[idx] = obst[0]