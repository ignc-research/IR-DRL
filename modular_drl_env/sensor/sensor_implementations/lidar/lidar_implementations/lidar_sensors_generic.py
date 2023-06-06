from modular_drl_env.robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from ..lidar import LidarSensor
from modular_drl_env.util.misc import regular_equidistant_sphere_points, fibonacci_sphere
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from time import time

class LidarSensorGeneric(LidarSensor):
    """
    Implements a generic lidar setup that will send 
    """

    def __init__(self, 
                 robot: Robot,
                 indicator_buckets: int,
                 ray_start: float, 
                 ray_end: float, 
                 ray_setup: dict, 
                 sim_step: float,
                 sim_steps_per_env_step: int,
                 indicator: bool=True,
                 ignore_self: bool=True,
                 report_points: bool=False,
                 normalize: bool=False,
                 add_to_observation_space: bool=True, 
                 add_to_logging: bool=False, 
                 update_steps: int=1
                 ):
        super().__init__(robot, indicator_buckets, sim_step, sim_steps_per_env_step, indicator, normalize, add_to_observation_space, add_to_logging, update_steps)

        # this is a dict which will contain link names as keys and number of rays distributed in a sphere around the link as value
        self.ray_setup = ray_setup

        # lidar ray lengths
        self.ray_start = ray_start
        self.ray_end = ray_end

        # indicator conversion setup
        raw_bucket_size = 1 / indicator_buckets  # 1 is the range of pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2 / indicator_buckets  # 2 is the range of the indicator data (from -1 to 1)
        # lambda function to convert to indicator based on bucket size
        self.raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

        # determine shape of out output and the direction of rays
        # INFO: because we won't rotate the rays in this class, we can already calculate every single ray right here and then later on just add the offset
        # of the world frame location that the respective link is at
        # this also allows us to calculate the shape of the output, because sometimes our circle point algorithm won't be able to return the exact number of points
        # the user specified as the amount of the rays at the start (usually it's at most 2 off)
        self.lidar_indicator_shape = 0
        self.rays_starts_base = {}
        self.rays_ends_base  = {}
        for link in self.ray_setup:
            #sampled_sphere_points = regular_equidistant_sphere_points(self.ray_setup[link], self.ray_end)
            sampled_sphere_points = fibonacci_sphere(self.ray_setup[link]) * self.ray_end
            self.lidar_indicator_shape += len(sampled_sphere_points)
            self.rays_ends_base[link] = sampled_sphere_points
            self.rays_starts_base[link] = (sampled_sphere_points * self.ray_start / self.ray_end) # == the points where our rays are supposed to start

        # data storage for later
        self.rays_starts = []
        self.rays_ends = []
        self.results = []

        # bool for whether self hits get reported or treated as nothing hit
        self.ignore_self = ignore_self

        # bool for reporting the poinst where lidar rays hit something as xyz tuples in relation to the link center they originate from
        # this will x3 the size of the output
        # rays that don't hit anything will be reported as a hit at the maximum length (i.e. the output will be the point at the end of the ray)
        self.report_points = report_points
        self.endpoints = None

    def get_observation_space_element(self) -> dict:
        return {self.output_name: Box(low=-1, high=1, shape=(self.lidar_indicator_shape * (1 if not self.report_points else 3),), dtype=np.float32)}
    
    def _get_lidar_data(self):
        self.rays_starts = np.empty((0,3))
        self.rays_ends = np.empty((0,3))

        for link in self.ray_setup:
            pos, _, _, _ = pyb_u.get_link_state(self.robot.object_id, link)
            self.rays_starts = np.vstack([self.rays_starts, self.rays_starts_base[link] + pos])
            self.rays_ends = np.vstack([self.rays_ends, self.rays_ends_base[link] + pos])

        self.results = pyb.rayTestBatch(self.rays_starts, self.rays_ends)
        
        processed_results = np.array(self.results, dtype=object)
        if self.ignore_self:
            # set the hit fraction to 1 for all rays that hit the robot itself
            self_mask = processed_results[:, 0] == pyb_u.to_pb(self.robot.object_id)
            processed_results[self_mask][:, 2] = 1

        return processed_results[:,2]

    def _process_raw_lidar(self, raw_lidar_data):

        indicator = np.zeros(self.lidar_indicator_shape)
        distances = np.zeros(self.lidar_indicator_shape)
        if self.report_points:
            endpoints = np.zeros((self.lidar_indicator_shape, 3))
        else:
            endpoints = None

        for idx, data_point in enumerate(raw_lidar_data):
            indicator[idx] = self.raw_to_indicator(data_point)
            distances[idx] = data_point * (self.ray_end - self.ray_start) + self.ray_start
            if self.report_points:
                endpoints[idx] = self.rays_starts[idx] + data_point * (self.rays_ends[idx] - self.rays_starts[idx])

        return indicator, distances, endpoints

    def build_visual_aux(self):
        hitRayColor = [0, 1, 0]
        missRayColor = [1, 0, 0]

        for index, result in enumerate(self.results):
            if result[0] == -1:
                self.aux_lines += pyb_u.draw_lines([self.rays_starts[index]], [self.rays_ends[index]], [missRayColor])
            else:
                self.aux_lines += pyb_u.draw_lines([self.rays_starts[index]], [self.rays_ends[index]], [hitRayColor])

    def update(self, step):

        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            lidar_data_raw = self._get_lidar_data()
            self.lidar_indicator, self.lidar_distances, self.endpoints = self._process_raw_lidar(lidar_data_raw)    
        self.cpu_time = time() - self.cpu_epoch
        
        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        lidar_data_raw = self._get_lidar_data()
        self.lidar_indicator, self.lidar_distances, self.endpoints = self._process_raw_lidar(lidar_data_raw)
        self.cpu_time = time() - self.cpu_epoch
        self.aux_visual_objects = []

    def get_observation(self) -> dict:
        if self.report_points:
            return {self.output_name: np.concatenate(self.endpoints, axis=0)}
        if self.indicator:
            return {self.output_name: self.lidar_indicator}
        else:
            return {self.output_name: self.lidar_distances}