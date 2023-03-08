import pybullet as pyb
from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor
from modular_drl_env.robot.robot import Robot
from time import process_time

__all__ = [
    "ObstacleSensor"
]

class ObstacleSensor(Sensor):

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot: Robot, num_obstacles: int, max_distance: float, reference_link_id: int):
        
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps)

        # set associated robot
        self.robot = robot

        # num of obstacles reported in the output
        self.num_obstacles = num_obstacles
        # maximum distance which the sensor will consider for obstacles
        self.max_distance = max_distance
        # default observation
        # 0 0 0 vector and max distance, gets used when there are not enough obstacles in the env currently to fill out the observation
        self.default_observation = np.array([[0, 0, 0, self.max_distance] for _ in range(self.num_obstacles)], dtype=np.float32).flatten()
        # link id of the robot part in reference to which the distances and vectors will be reported
        self.reference_link_id = reference_link_id

        # set output data field name
        self.output_name = "nearest_" + str(self.num_obstacles) + "_obstacles_link_" + str(self.reference_link_id) + "_" + self.robot.name
        self.output_name_time = "obstacle_sensor_link_" + str(self.reference_link_id) + "_cpu_time_" + self.robot.name

        # init data storage
        self.output_vector = None
        self.data_raw = None

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / (np.ones(4 * self.num_obstacles) * self.max_distance * 2)
        self.normalizing_constant_b = np.ones(4 * self.num_obstacles) - np.multiply(self.normalizing_constant_a, np.ones(4 * self.num_obstacles) * self.max_distance)

    def update(self, step) -> dict:
        self.cpu_epoch = process_time()
        if step % self.update_steps == 0:
            self.output_vector = self.default_observation
            self.data_raw = self._run_obstacle_detection()
            new_data = self._process(self.data_raw)
            self.output_vector[:len(new_data)] = new_data
        self.cpu_time = process_time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = process_time()
        self.output_vector = self.default_observation
        self.data_raw = self._run_obstacle_detection()
        new_data = self._process(self.data_raw)
        self.output_vector[:len(new_data)] = new_data
        self.cpu_time = process_time() - self.cpu_epoch

    def get_observation(self) -> dict:
        if self.normalize:
            return self._normalize()
        else:
            return {self.output_name: self.output_vector}

    def _normalize(self) -> dict:
        return {self.output_name: np.multiply(self.normalizing_constant_a, self.output_vector) + self.normalizing_constant_b}

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            if self.normalize:
                return {self.output_name: Box(low=-1, high=1, shape=(4 * self.num_obstacles,), dtype=np.float32)}
            else:
                return {self.output_name: Box(low=-self.max_distance, high=self.max_distance, shape=(4 * self.num_obstacles,), dtype=np.float32)}
        else:
            return {}

    def _run_obstacle_detection(self):

        res = []
        # get nearest robots
        for robot in self.robot.world.robots_in_world:
            if robot.object_id != self.robot.object_id:
                closestPoints = pyb.getClosestPoints(self.robot.object_id, robot.object_id, self.max_distance, self.reference_link_id)
                min_val = min(closestPoints, key=lambda x: x[8])  # index 8 is the distance in the object returned by pybullet
                res.append(np.hstack([np.array(min_val[5]), np.array(min_val[6]), min_val[8]]))  # start, end, distance
        # get nearest obstacles
        for obstacle_id in self.robot.world.objects_ids:
            closestPoints = pyb.getClosestPoints(self.robot.object_id, obstacle_id, self.max_distance, self.reference_link_id)
            min_val = min(closestPoints, key=lambda x: x[8])
            res.append(np.hstack([np.array(min_val[5]), np.array(min_val[6]), min_val[8]]))
        # sort
        res.sort(key=lambda x: x[6])
        # extract n closest ones
        smallest_n = res[:self.num_obstacles]

        return np.array(smallest_n)

    def _process(self, data_raw):
        data_processed = []
        for i in range(len(data_raw)):
            data_processed.append(np.hstack([data_raw[i][3:6] -  data_raw[i][0:3], data_raw[i][6]]))
        return np.array(data_processed).flatten()

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict[self.output_name] = self.output_vector
        logging_dict[self.output_name_time] = self.cpu_time

        return logging_dict

    def build_visual_aux(self):
        position = np.array(pyb.getLinkState(self.robot.object_id, self.reference_link_id)[4])

        line_starts = [self.data_raw[i][0:3] for i in range(len(self.data_raw))]
        line_ends = [self.data_raw[i][3:6] for i in range(len(self.data_raw))]

        for i in range(len(line_starts)):
            self.aux_visual_objects.append(pyb.addUserDebugLine(line_starts[i], line_ends[i], [0, 0, 1]))
