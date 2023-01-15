from sensor.sensor import Sensor
from robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from time import time
from abc import abstractmethod

class LidarSensor(Sensor):
    """
    Base class for a lidar sensor.
    Must be subclassed for each robot because the setup of the rays changes.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, robot:Robot, indicator_buckets:int, render:bool=False, indicator:bool=True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step)

        # set associated robot
        self.robot = robot

        # bool for whether the rays get rendered
        self.render = render

        # bool for wether the return data is the distances measured by the lidar or a processed indicator
        self.indicator = indicator

        # name for the observation space element
        self.output_name = "lidar_" + self.robot.name

        # number of buckets for the indicator (e.g. 2 would mean indicator values could -1 and 1, 3 would mean -1, 0 and 1 etc.)
        self.indicator_buckets = indicator_buckets

        # data storage
        self.lidar_indicator = None
        self.lidar_distances = None

    def update(self):

        self.cpu_epoch = time()
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


class LidarSensorUR5(LidarSensor):
    """
    Lidar class adapted for the use with the UR5. Features rays coming from the end effector and several wrist links.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, robot: Robot, indicator_buckets:int, ray_start: float, ray_end: float, num_rays_side: int, num_rays_circle_directions: int, render: bool = False, indicator: bool = True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, robot, indicator_buckets, render, indicator)

        # lidar setup attributes
        self.ray_start = ray_start  # offset of the ray start from the mesh center
        self.ray_end = ray_end  # end of the ray, meaning ray length = ray_end - ray_start
        self.num_rays_circle_directions = num_rays_circle_directions  # number of directions that the circle is divided into for the sideways rays
        self.num_rays_side = num_rays_side  # rays to cast per sideways direction

    def get_observation_space_element(self) -> dict:
        return {self.output_name: Box(low=-1, high=1, shape=(1 + 4 * self.num_rays_circle_directions,), dtype=np.float32)}

    def _get_lidar_data(self):
        
        rays_starts = []
        rays_ends = []

        # get link states
        # link IDs hardcoded for the URDF file we use
        linkState_ee = pyb.getLinkState(self.robot.object_id, 7)
        linkState_wrist1 = pyb.getLinkState(self.robot.object_id, 4)
        linkState_wrist2 = pyb.getLinkState(self.robot.object_id, 5)
        linkState_wrist3 = pyb.getLinkState(self.robot.object_id, 6)
        linkState_arm3 = pyb.getLinkState(self.robot.object_id, 3)

        # create frame matrices
        frame_ee = np.eye(4)
        frame_ee[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_ee[5]), (3,3))
        frame_ee[0:3, 3] = linkState_ee[4]
        frame_wrist1 = np.eye(4)
        frame_wrist1[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist1[5]), (3,3))
        frame_wrist1[0:3, 3] = linkState_wrist1[4]
        frame_wrist2 = np.eye(4)
        frame_wrist2[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist2[5]), (3,3))
        frame_wrist2[0:3, 3] = linkState_wrist2[4]
        frame_wrist3 = np.eye(4)
        frame_wrist3[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist3[5]), (3,3))
        frame_wrist3[0:3, 3] = linkState_wrist3[4]
        frame_arm3 = np.eye(4)
        frame_arm3[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_arm3[5]), (3,3))
        frame_arm3[0:3, 3] = linkState_arm3[4]

        # add the ray that goes straight forward out of the end effector
        rays_starts.append(linkState_ee[4])
        rays_ends.append(np.matmul(frame_ee, np.array([0, 0, self.ray_end, 1]).T)[0:3].tolist())

        # run through each frame to add ray starts and ends
        for angle in np.linspace(-np.pi/2, np.pi/2, self.num_rays_circle_directions):
            for i in range(self.num_rays_side):
                interval = 0.01
                rays_starts.append(np.matmul(frame_wrist3, np.array([0.0, i * interval - 0.05, 0.0, 1]).T)[0:3].tolist())
                rays_ends.append(np.matmul(frame_wrist3, np.array([self.ray_end * np.sin(angle), i * interval - 0.05, self.ray_end * np.cos(angle), 1]).T)[0:3].tolist())
        for angle in np.linspace(-np.pi/2, np.pi/2, self.num_rays_circle_directions):
            for i in range(self.num_rays_side):
                # TODO: this does not seem to work for all orientations of the UR5 robot
                # at some angles, the rays of this wrist will all point towards the inside
                # this doesn't happen in the default experiments, but might become acute if other experiments use different poses
                interval = 0.01
                rays_starts.append(np.matmul(frame_wrist2, np.array([0.0, 0.0, i * interval - 0.03, 1]).T)[0:3].tolist())
                rays_ends.append(np.matmul(frame_wrist2, np.array([-self.ray_end * np.cos(angle), self.ray_end * np.sin(angle), i * interval - 0.03, 1]).T)[0:3].tolist())
        for angle in np.linspace(-np.pi/2, np.pi/2, self.num_rays_circle_directions):
            for i in range(self.num_rays_side):
                interval = 0.01
                rays_starts.append(np.matmul(frame_wrist1, np.array([0.0, i * interval - 0.03, 0.0, 1]).T)[0:3].tolist())
                rays_ends.append(np.matmul(frame_wrist1, np.array([self.ray_end * np.sin(angle), i * interval - 0.03, self.ray_end * np.cos(angle), 1]).T)[0:3].tolist())
        for angle in np.linspace(-3*np.pi/4, np.pi, self.num_rays_circle_directions):
            for i in range(self.num_rays_side):
                interval = 0.02
                rays_starts.append(np.matmul(frame_arm3, np.array([0.0, 0.0, i * interval + 0.1, 1]).T)[0:3].tolist())
                rays_ends.append(np.matmul(frame_arm3, np.array([self.ray_end * np.sin(angle), -self.ray_end * np.cos(angle), i * interval + 0.1, 1]).T)[0:3].tolist())

        results = pyb.rayTestBatch(rays_starts, rays_ends)
        
        if self.render:
            hitRayColor = [0, 1, 0]
            missRayColor = [1, 0, 0]

            pyb.removeAllUserDebugItems()  # this will kill workspace borders if they are displayed 

            for index, result in enumerate(results):
                if result[0] == -1:
                    pyb.addUserDebugLine(rays_starts[index], rays_ends[index], missRayColor)
                else:
                    pyb.addUserDebugLine(rays_starts[index], rays_ends[index], hitRayColor)
        
        return np.array(results, dtype=object)[:,2]  # keeps only the distance information

    def _process_raw_lidar(self, raw_lidar_data):
        raw_bucket_size = 1 / self.indicator_buckets  # 1 is the range of pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2 / self.indicator_buckets  # 2 is the range of the indicator data (from -1 to 1)
        
        # lambda function to convert to indicator based on bucket size
        raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

        lidar_shape = 1 + 4 * self.num_rays_circle_directions 
        indicator = np.zeros(lidar_shape)
        distances = np.zeros(lidar_shape)

        # conversion
        # the list slicing here is messy but basically just follows the way the rays were put into the source array following the _get_lidar_data method
        # tip
        indicator[0] = raw_to_indicator(raw_lidar_data[0])
        distances[0] = raw_lidar_data[0] * (self.ray_end - self.ray_start) + self.ray_start
        # wrist 3
        for i in range(self.num_rays_circle_directions):
            lidar_min = raw_lidar_data[1 + i * self.num_rays_side : 1 + (i + 1) * self.num_rays_side].min()
            indicator[1 + i] = raw_to_indicator(lidar_min)
            distances[1 + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
        # wrist 2
        for i in range(self.num_rays_circle_directions):
            lidar_min = raw_lidar_data[1 + self.num_rays_circle_directions * self.num_rays_side + i * self.num_rays_side : 1 + self.num_rays_circle_directions * self.num_rays_side + (i + 1) * self.num_rays_side].min()
            indicator[1 + self.num_rays_circle_directions + i] = raw_to_indicator(lidar_min)
            distances[1 + self.num_rays_circle_directions + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
        # wrist 1
        for i in range(self.num_rays_circle_directions):
            lidar_min = raw_lidar_data[1 + 2 * self.num_rays_circle_directions * self.num_rays_side + i * self.num_rays_side : 1 + 2 * self.num_rays_circle_directions * self.num_rays_side + (i + 1) * self.num_rays_side].min()
            indicator[1 + 2 * self.num_rays_circle_directions + i] = raw_to_indicator(lidar_min)
            distances[1 + 2 * self.num_rays_circle_directions + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
        # arm 3
        for i in range(self.num_rays_circle_directions):
            lidar_min = raw_lidar_data[1 + 3 * self.num_rays_circle_directions * self.num_rays_side + i * self.num_rays_side : 1 + 3 * self.num_rays_circle_directions * self.num_rays_side + (i + 1) * self.num_rays_side].min()
            indicator[1 + 3 * self.num_rays_circle_directions + i] = raw_to_indicator(lidar_min)
            distances[1 + 3 * self.num_rays_circle_directions + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
    
        return indicator, distances