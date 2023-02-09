import pybullet as pyb
from gym.spaces import Box
import numpy as np
from sensor.sensor import Sensor
from robot.robot import Robot
from time import process_time

__all__ = [
    "ObstacleSensor"
]

class ObstacleSensor(Sensor):

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot: Robot, num_obstacles: int, max_distance: float):
        
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

        # set output data field name
        self.output_name = "nearest_" + str(self.num_obstacles) + "_obstacles_" + self.robot.name
        self.output_name_time = "obstacle_sensor_cpu_time_" + self.robot.name

        # init data storage
        self.output_vector = None

        # normalizing constants for faster normalizing
        self.normalizing_constant_a = 2 / (np.ones(4 * self.num_obstacles) * self.max_distance * 2)
        self.normalizing_constant_b = np.ones(4 * self.num_obstacles) - np.multiply(self.normalizing_constant_a, np.ones(4 * self.num_obstacles) * self.max_distance)

    def update(self, step) -> dict:
        self.cpu_epoch = process_time()
        if step % self.update_steps == 0:
            self.output_vector = self.default_observation
            new_data = self._run_obstacle_detection()
            self.output_vector[:len(new_data)] = new_data
        self.cpu_time = process_time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = process_time()
        self.output_vector = self.default_observation
        new_data = self._run_obstacle_detection()
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
                closestPoints = pyb.getClosestPoints(self.robot.object_id, robot.object_id, self.max_distance, self.robot.end_effector_link_id)
                min_val = min(closestPoints, key=lambda x: x[8])  # index 8 is the distance in the object returned by pybullet
                res.append(np.hstack([-np.array(min_val[7]), min_val[8]]))  # direction vector and distance
        # get nearest obstacles
        for obstacle_id in self.robot.world.objects_ids:
            closestPoints = pyb.getClosestPoints(self.robot.object_id, obstacle_id, self.max_distance, self.robot.end_effector_link_id)
            min_val = min(closestPoints, key=lambda x: x[8])
            res.append(np.hstack([-np.array(min_val[7]), min_val[8]]))
        # sort
        res.sort(key=lambda x: x[3])
        # extract n closest ones
        smallest_n = res[:self.num_obstacles]

        """
        line_starts = [self.robot.position_rotation_sensor.position for i in range(self.num_obstacles)]
        line_ends = [line_starts[i] + (ele[3] * ele[0:3]) for i, ele in enumerate(smallest_n)]
        for i in range(len(line_starts)):
            pyb.addUserDebugLine(line_starts[i], line_ends[i], [0, 0, 1])
        """

        return np.array(smallest_n).flatten()

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict[self.output_name] = self.output_vector
        logging_dict[self.output_name_time] = self.cpu_time

        return logging_dict
