from sensor.sensor import Sensor
from robot.robot import Robot
from time import time
from abc import abstractmethod

class LidarSensor(Sensor):
    """
    Base class for a lidar sensor.
    Must be subclassed for each robot because the setup of the rays changes.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot:Robot, indicator_buckets:int, indicator:bool=True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps)

        # set associated robot
        self.robot = robot

        # bool for wether the return data is the distances measured by the lidar or a processed indicator
        self.indicator = indicator

        # name for the observation space element
        self.output_name = "lidar_" + self.robot.name

        # number of buckets for the indicator (e.g. 2 would mean indicator values could -1 and 1, 3 would mean -1, 0 and 1 etc.)
        self.indicator_buckets = indicator_buckets

        # data storage
        self.lidar_indicator = None
        self.lidar_distances = None

    def update(self, step):

        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            lidar_data_raw = self._get_lidar_data()
            self.lidar_indicator, self.lidar_distances = self._process_raw_lidar(lidar_data_raw)    
        self.cpu_time = time() - self.cpu_epoch
        
        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        lidar_data_raw = self._get_lidar_data()
        self.lidar_indicator, self.lidar_distances = self._process_raw_lidar(lidar_data_raw)
        self.cpu_time = time() - self.cpu_epoch

    def get_observation(self) -> dict:
        if self.indicator:
            return {self.output_name: self.lidar_indicator}
        else:
            return {self.output_name: self.lidar_distances}

    def _normalize(self) -> dict:
        pass  # the way we construct the lidar data it will always be normalized

    def get_data_for_logging(self) -> dict:
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict["lidar_indicator_" + self.robot.name] = self.lidar_indicator
        logging_dict["lidar_distances_" + self.robot.name] = self.lidar_distances
        logging_dict["lidar_sensor_cpu_time_" + self.robot.name] = self.cpu_time

        return logging_dict

    @abstractmethod
    def _get_lidar_data(self):
        """
        This should implement the concrete PyBullet call to raycasting adapted to a specific robot.
        """
        pass

    @abstractmethod
    def _process_raw_lidar(self, raw_lidar_data):
        """
        This should implement a conversion of raw PyBullet raycasting data to an indicator with values from -1 to 1
        and distances between 0 and 1 (meaning that maximum ray length should not be larger than 1)
        """
        pass


