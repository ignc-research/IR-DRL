from robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from ..lidar import LidarSensor

__all__ = [
    'LidarSensorKR16'
]

class LidarSensorKR16(LidarSensor):

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot: Robot, indicator_buckets:int, ray_start: float, ray_end: float, ray_setup: dict, indicator: bool = True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, robot, indicator_buckets, indicator)

        # dict which governs which robot links get lidar rays and how many
        # possible keys:
        # ee_forward, ee_cone, ee_side_circle, ee_back_cone, upper_arm
        # each key (with exception of ee_forward, because it's always just one ray) has a list with two elements
        # first entry is num of directions, second entry is num of rays per direction
        self.ray_setup = ray_setup

        # lidar ray lengths
        self.ray_start = ray_start  # offset of the ray start from the mesh center
        self.ray_end = ray_end  # end of the ray, meaning ray length = ray_end - ray_start

        # indicator conversion setup
        raw_bucket_size = 1 / indicator_buckets  # 1 is the range of pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2 / indicator_buckets  # 2 is the range of the indicator data (from -1 to 1)
        # lambda function to convert to indicator based on bucket size
        self.raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

        # determine shape out output
        self.lidar_indicator_shape = 0
        for key in self.ray_setup:
            if key == "ee_forward":
                self.lidar_indicator_shape += 1
                self.ray_setup[key] = [1, 1]  # overwrite to be safe
            else:
                self.lidar_indicator_shape += self.ray_setup[key][0]

        # because a dict can be ordered arbitrarily, we need to record how the code below puts lidar results in the output ...
        self.output_order = ["ee_forward", "ee_cone", "ee_side_circle", "ee_back_cone", "upper_arm"]
        # and then keep only those that were activated by the user, this way we can process the results from the PyBullet call in the right order
        self.output_order = [ele for ele in self.output_order if ele in self.ray_setup]

        # data storage
        self.rays_starts = []
        self.rays_ends = []
        self.results = []


    def get_observation_space_element(self) -> dict:
        return {self.output_name: Box(low=-1, high=1, shape=(self.lidar_indicator_shape,), dtype=np.float32)}

    def _get_lidar_data(self):
        
        self.rays_starts = []
        self.rays_ends = []

        
        linkState_ee = pyb.getLinkState(self.robot.object_id, 6)
        frame_ee = np.eye(4)
        frame_ee[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_ee[5]), (3,3))
        frame_ee[0:3, 3] = linkState_ee[4]

        # deal with all activated links
        # end effector forwards ray:
        if "ee_forward" in self.ray_setup:
            self.rays_starts.append(np.matmul(frame_ee, np.array([0, 0, self.ray_start, 1]).T)[0:3].tolist())
            self.rays_ends.append(np.matmul(frame_ee, np.array([0, 0, self.ray_end, 1]).T)[0:3].tolist())

        interval = -0.005
        # cone rays from end effector
        if "ee_cone" in self.ray_setup:
            for angle in np.linspace(230 * (np.pi / 180), 270 * (np.pi/180), self.ray_setup["ee_cone"][0]):
                for i in range(self.ray_setup["ee_cone"][1]):
                    z = -self.ray_end * np.sin(angle)
                    l = self.ray_end * np.cos(angle)
                    x_end = l * np.cos(2 * np.pi * i / self.ray_setup["ee_cone"][1])
                    y_end = l * np.sin(2 * np.pi * i / self.ray_setup["ee_cone"][1])
                    self.rays_starts.append(np.matmul(frame_ee, np.array([0, 0, self.ray_start, 1]).T)[0:3].tolist())
                    self.rays_ends.append(np.matmul(frame_ee, np.array([x_end, y_end, z, 1]).T)[0:3].tolist())
        
        # around head circle rays
        if "ee_side_circle" in self.ray_setup:
            for angle in np.linspace(0, 2 * np.pi - 2 * np.pi/self.ray_setup["ee_side_circle"][0], self.ray_setup["ee_side_circle"][0]):
                for i in range(self.ray_setup["ee_side_circle"][1]):
                    z_start = i * interval - 0.1
                    x_start = self.ray_start * np.cos(angle)
                    y_start = self.ray_start * np.sin(angle)
                    z_end = i * interval - 0.1
                    x_end = self.ray_end * np.cos(angle)
                    y_end = self.ray_end * np.sin(angle)
                    self.rays_starts.append(np.matmul(frame_ee, np.array([x_start, y_start, z_start, 1]).T)[0:3].tolist())
                    self.rays_ends.append(np.matmul(frame_ee, np.array([x_end, y_end, z_end, 1]).T)[0:3].tolist())

        # rays from the back of the end effector
        if "ee_back_cone" in self.ray_setup:
            for angle in np.linspace(230 * (np.pi / 180), 290 * (np.pi/180), self.ray_setup["ee_back_cone"][0]):
                for i in range(self.ray_setup["ee_back_cone"][1]):
                    l = self.ray_end * np.cos(angle)
                    z_end = self.ray_end * np.sin(angle)
                    x_end = l * np.cos(np.pi * i / self.ray_setup["ee_back_cone"][1] - np.pi / 2)
                    y_end = l * np.sin(np.pi * i / self.ray_setup["ee_back_cone"][1] - np.pi / 2)
                    self.rays_starts.append(np.matmul(frame_ee, np.array([0, 0, self.ray_start - 0.25, 1]).T)[0:3].tolist())
                    self.rays_ends.append(np.matmul(frame_ee, np.array([x_end, y_end, z_end - 0.25, 1]).T)[0:3].tolist())

        # rays around the upper arm of the end effector
        if "upper_arm" in self.ray_setup: 
            interval = -0.48 / self.ray_setup["upper_arm"][1]  # evenly space rays along entire length, arm length of 0.48 found out by testing and does not account for potential urdf mesh scaling
            linkState_arm = pyb.getLinkState(self.robot.object_id, 2)
            frame_arm = np.eye(4)
            frame_arm[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_arm[5]), (3,3))
            frame_arm[0:3, 3] = linkState_arm[4]
            extra_offset = 0.095
            for angle in np.linspace(0, 2 * np.pi - 2 * np.pi/self.ray_setup["upper_arm"][0], self.ray_setup["upper_arm"][0]):
                for i in range(self.ray_setup["upper_arm"][1]):
                    self.rays_starts.append(np.matmul(frame_arm, np.array([i * interval + 0.5, (self.ray_start + extra_offset) * np.sin(angle), (-self.ray_start - extra_offset) * np.cos(angle) - 0.05, 1]).T)[0:3].tolist())
                    self.rays_ends.append(np.matmul(frame_arm, np.array([i * interval + 0.5, self.ray_end * np.sin(angle), -self.ray_end * np.cos(angle) - 0.05, 1]).T)[0:3].tolist())

        self.results = pyb.rayTestBatch(self.rays_starts, self.rays_ends)
        
        return np.array(self.results, dtype=object)[:,2]  # keeps only the distance information

    def _process_raw_lidar(self, raw_lidar_data):

        indicator = np.zeros(self.lidar_indicator_shape)
        distances = np.zeros(self.lidar_indicator_shape)

        # conversion
        # the list slicing here is messy but basically just follows the way the rays were put into the source array following the _get_lidar_data method
        raw_data_offset = 0
        indicator_offset = 0
        for j in range(len(self.output_order)): 
            # get nums for this link
            n_rays_p_direction = self.ray_setup[self.output_order[j]][1]
            for i in range(self.ray_setup[self.output_order[j]][0]):
                lidar_min = raw_lidar_data[raw_data_offset + i * n_rays_p_direction : raw_data_offset + (i + 1) * n_rays_p_direction].min()
                indicator[indicator_offset + i] = self.raw_to_indicator(lidar_min)
                distances[indicator_offset + i] = lidar_min * (self.ray_end - self.ray_start) + self.ray_start
            raw_data_offset += n_rays_p_direction * self.ray_setup[self.output_order[j]][0]
            indicator_offset += self.ray_setup[self.output_order[j]][0]
    
        return indicator, distances

    def build_visual_aux(self):
        hitRayColor = [0, 1, 0]
        missRayColor = [1, 0, 0]

        for index, result in enumerate(self.results):
            if result[0] == -1:
                self.aux_visual_pyb_objects.append(pyb.addUserDebugLine(self.rays_starts[index], self.rays_ends[index], missRayColor))
            else:
                self.aux_visual_pyb_objects.append(pyb.addUserDebugLine(self.rays_starts[index], self.rays_ends[index], hitRayColor))