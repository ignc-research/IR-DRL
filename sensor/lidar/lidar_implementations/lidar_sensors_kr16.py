from robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from ..lidar import LidarSensor

__all__ = [
    'LidarSensorKR16'
]

class LidarSensorKR16(LidarSensor):

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot: Robot, indicator_buckets:int, ray_start: float, ray_end: float, num_rays_side: int, num_rays_circle_directions: int, activated_links=[0,1,2,3,4], render: bool = False, indicator: bool = True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, robot, indicator_buckets, render, indicator)

        # lidar setup attributes
        self.ray_start = ray_start  # offset of the ray start from the mesh center
        self.ray_end = ray_end  # end of the ray, meaning ray length = ray_end - ray_start
        self.num_rays_circle_directions = num_rays_circle_directions  # number of directions that the circle is divided into for the sideways rays
        self.num_rays_side = num_rays_side  # rays to cast per sideways direction

        # indicator conversion setup
        raw_bucket_size = 1 / indicator_buckets  # 1 is the range of pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2 / indicator_buckets  # 2 is the range of the indicator data (from -1 to 1)
        # lambda function to convert to indicator based on bucket size
        self.raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

        # set which parts of the lidar are being used
        # usage: 
        # 0: end effector forward ray
        # 1: cone of rays around end effector
        # 2: circle of around end effector wrist with slight offset
        self.activated_links = np.zeros(3)
        for link in activated_links:
            self.activated_links[link] = 1

        self.lidar_indicator_shape = int(self.activated_links[0] + np.sum(self.activated_links[1:]) * self.num_rays_circle_directions)

    def get_observation_space_element(self) -> dict:
        return {self.output_name: Box(low=-1, high=1, shape=(self.lidar_indicator_shape,), dtype=np.float32)}

    def _get_lidar_data(self):
        
        rays_starts = []
        rays_ends = []

        
        linkState_wrist1 = pyb.getLinkState(self.robot.object_id, 3)
        linkState_wrist2 = pyb.getLinkState(self.robot.object_id, 4)
        linkState_wrist3 = pyb.getLinkState(self.robot.object_id, 5)
        linkState_arm3 = pyb.getLinkState(self.robot.object_id, 2)

        linkState_ee = pyb.getLinkState(self.robot.object_id, 6)
        frame_ee = np.eye(4)
        frame_ee[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_ee[5]), (3,3))
        frame_ee[0:3, 3] = linkState_ee[4]

        # deal with all activated links
        # end effector forwards ray:
        if self.activated_links[0]:
            rays_starts.append(np.matmul(frame_ee, np.array([0, 0, self.ray_start, 1]).T)[0:3].tolist())
            rays_ends.append(np.matmul(frame_ee, np.array([0, 0, self.ray_end, 1]).T)[0:3].tolist())

        # cone rays from end effector
        if self.activated_links[1]:

            for angle in np.linspace(230 * (np.pi / 180), 270 * (np.pi/180), self.num_rays_circle_directions):
                for i in range(self.num_rays_side):
                    z = -self.ray_end * np.sin(angle)
                    l = self.ray_end * np.cos(angle)
                    x_end = l * np.cos(2 * np.pi * i / self.num_rays_side)
                    y_end = l * np.sin(2 * np.pi * i / self.num_rays_side)
                    rays_starts.append(np.matmul(frame_ee, np.array([0, 0, self.ray_start, 1]).T)[0:3].tolist())
                    rays_ends.append(np.matmul(frame_ee, np.array([x_end, y_end, z, 1]).T)[0:3].tolist())
        
        # around head circle rays
        if self.activated_links[2]:

            interval = -0.005
            for angle in np.linspace(0, 2 * np.pi - 2 * np.pi/self.num_rays_circle_directions, self.num_rays_circle_directions):
                for i in range(self.num_rays_side):
                    z_start = i * interval-0.1
                    x_start = self.ray_start * np.cos(angle)
                    y_start = self.ray_start * np.sin(angle)
                    z_end = i * interval-0.1
                    x_end = self.ray_end * np.cos(angle)
                    y_end = self.ray_end * np.sin(angle)
                    rays_starts.append(np.matmul(frame_ee, np.array([x_start,y_start,z_start,1]).T)[0:3].tolist())
                    rays_ends.append(np.matmul(frame_ee, np.array([x_end,y_end,z_end,1]).T)[0:3].tolist())

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

        indicator = np.zeros(self.lidar_indicator_shape)
        distances = np.zeros(self.lidar_indicator_shape)

        # conversion
        # the list slicing here is messy but basically just follows the way the rays were put into the source array following the _get_lidar_data method
        # tip
        if self.activated_links[0]:
            indicator[0] = self.raw_to_indicator(raw_lidar_data[0])
            distances[0] = raw_lidar_data[0] * (self.ray_end - self.ray_start) + self.ray_start
        # wrist 3
        for j in range(int(np.sum(self.activated_links[1:]))): # wrist 3, wrist 2, wrist 1, arm 3
            for i in range(self.num_rays_circle_directions):
                lidar_min = raw_lidar_data[1 + j * self.num_rays_circle_directions * self.num_rays_side + i * self.num_rays_side : 1 + j * self.num_rays_circle_directions * self.num_rays_side + (i + 1) * self.num_rays_side].min()
                indicator[1 + j * self.num_rays_circle_directions + i] = self.raw_to_indicator(lidar_min)
                distances[1 + j * self.num_rays_circle_directions + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
    
        return indicator, distances