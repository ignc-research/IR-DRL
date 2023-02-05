from robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from time import time
from abc import abstractmethod
from ..lidar import LidarSensor


__all__ = [
    'LidarSensorKukaiiwa'

]

class LidarSensorKukaiiwa(LidarSensor):
    """
    Lidar class adapted for the use with the UR5. Features rays coming from the end effector and several wrist links.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, robot: Robot, indicator_buckets:int, ray_start: float, ray_end: float, ray_setup: dict, render: bool = False, indicator: bool = True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps, robot, indicator_buckets, render, indicator)

        # dict which governs which robot links get lidar rays and how many
        # possible keys:
        # ee_forward, wrist3_circle, wrist2_circle, wrist1_circle, upper_arm
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
        self.output_order = ["ee_forward", "wrist3_circle", "wrist2_circle", "wrist1_circle", "upper_arm"]
        # and then keep only those that were activated by the user, this way we can process the results from the PyBullet call in the right order
        self.output_order = [ele for ele in self.output_order if ele in self.ray_setup]


    def get_observation_space_element(self) -> dict:
        return {self.output_name: Box(low=-1, high=1, shape=(self.lidar_indicator_shape,), dtype=np.float32)}

    def _get_lidar_data(self):
        
        rays_starts = []
        rays_ends = []

        # deal with all activated links:
        # get the local frames, then use a local definition of the rays to translate them into the global coordinate system
        # end effector forwards ray:
        if "ee_forward" in self.ray_setup:
            linkState_ee = pyb.getLinkState(self.robot.object_id, 6)
            frame_ee = np.eye(4)
            frame_ee[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_ee[5]), (3,3))
            frame_ee[0:3, 3] = linkState_ee[4]

            rays_starts.append(linkState_ee[4])
            rays_ends.append(np.matmul(frame_ee, np.array([0, 0, self.ray_end, 1]).T)[0:3].tolist())

        # wrist 3 rays (half circle)
        if "wrist3_circle" in self.ray_setup:
            linkState_wrist3 = pyb.getLinkState(self.robot.object_id, 5)
            frame_wrist3 = np.eye(4)
            frame_wrist3[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist3[5]), (3,3))
            frame_wrist3[0:3, 3] = linkState_wrist3[4]

            for angle in np.linspace(-np.pi/2, np.pi/2, self.ray_setup["wrist3_circle"][0]):
                for i in range(self.ray_setup["wrist3_circle"][1]):
                    interval = 0.01
                    rays_starts.append(np.matmul(frame_wrist3, np.array([0.0, i * interval - 0.05, 0.0, 1]).T)[0:3].tolist())
                    rays_ends.append(np.matmul(frame_wrist3, np.array([self.ray_end * np.sin(angle), i * interval - 0.05, self.ray_end * np.cos(angle), 1]).T)[0:3].tolist())
        
        # wrist 2 rays (half circle)
        if "wrist2_circle" in self.ray_setup:
            linkState_wrist2 = pyb.getLinkState(self.robot.object_id, 4)
            frame_wrist2 = np.eye(4)
            frame_wrist2[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist2[5]), (3,3))
            frame_wrist2[0:3, 3] = linkState_wrist2[4]

            for angle in np.linspace(-np.pi/2, np.pi/2, self.ray_setup["wrist3_circle"][0]):
                for i in range(self.ray_setup["wrist3_circle"][1]):
                    # TODO: this does not seem to work for all orientations of the UR5 robot
                    # at some angles, the rays of this wrist will all point towards the inside
                    # this doesn't happen in the default experiments, but might become acute if other experiments use different poses
                    interval = 0.01
                    rays_starts.append(np.matmul(frame_wrist2, np.array([0.0, 0.0, i * interval - 0.03, 1]).T)[0:3].tolist())
                    rays_ends.append(np.matmul(frame_wrist2, np.array([-self.ray_end * np.cos(angle), self.ray_end * np.sin(angle), i * interval - 0.03, 1]).T)[0:3].tolist())

        # wrist 1 rays (half circle)
        if "wrist1_circle" in self.ray_setup:
            linkState_wrist1 = pyb.getLinkState(self.robot.object_id, 3)
            frame_wrist1 = np.eye(4)
            frame_wrist1[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist1[5]), (3,3))
            frame_wrist1[0:3, 3] = linkState_wrist1[4]

            for angle in np.linspace(-np.pi/2, np.pi/2, self.ray_setup["wrist3_circle"][0]):
                for i in range(self.ray_setup["wrist3_circle"][1]):
                    interval = 0.01
                    rays_starts.append(np.matmul(frame_wrist1, np.array([0.0, i * interval - 0.03, 0.0, 1]).T)[0:3].tolist())
                    rays_ends.append(np.matmul(frame_wrist1, np.array([self.ray_end * np.sin(angle), i * interval - 0.03, self.ray_end * np.cos(angle), 1]).T)[0:3].tolist())

        # arm 3 rays (full circle)
        if "upper_arm" in self.ray_setup:
            linkState_arm3 = pyb.getLinkState(self.robot.object_id, 2)
            frame_arm3 = np.eye(4)
            frame_arm3[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_arm3[5]), (3,3))
            frame_arm3[0:3, 3] = linkState_arm3[4]

            for angle in np.linspace(-np.pi, np.pi - 2 * np.pi / self.ray_setup["upper_arm"][0], self.ray_setup["upper_arm"][0]):
                for i in range(self.ray_setup["upper_arm"][1]):
                    interval = 0.26 / self.ray_setup["upper_arm"][1]  # evenly space rays along entire length, arm length of 0.26 found out by testing and does not account for potential urdf mesh scaling
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



