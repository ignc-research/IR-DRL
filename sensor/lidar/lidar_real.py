from sensor.sensor import Sensor
from robot.robot import Robot
import numpy as np
import pybullet as pyb
from gym.spaces import Box
from time import time
from sensor.lidar import LidarSensor

class LidarSensorUR5Real(LidarSensor):
    """
    Lidar class adapted for the use with the UR5 with a realistic lidar setup. Features rays coming from a spot where a plausible lidar sensor could be mounted in real life.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, robot: Robot, indicator_buckets:int, ray_start: float, ray_end: float, render: bool = False, indicator: bool = True):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, robot, indicator_buckets, render, indicator)

        # lidar setup attributes
        self.ray_start = ray_start  # offset of the ray start from the mesh center
        self.ray_end = ray_end  # end of the ray, meaning ray length = ray_end - ray_start

    def get_observation_space_element(self) -> dict:
        #Define Output in Observation Space, if hard code is needes instead of self.num rays circle just input an integer
        #8 rays for ring, see: https://www.exp-tech.de/sensoren/lidar/9569/teraranger-tower-evo-600hz-8-sensors?c=1494
        return {self.output_name: Box(low=-1, high=1, shape=(8,), dtype=np.float32)}

    #sets direction of rays and calls pybullet for raycasting
    def _get_lidar_data(self):
        
        rays_starts = []
        rays_ends = []

        # get link states
        # link IDs hardcoded for the URDF file we use
        linkState_wrist3 = pyb.getLinkState(self.robot.object_id, 6)

        # create frame matrices
        frame_wrist3 = np.eye(4)
        frame_wrist3[:3, :3] = np.reshape(pyb.getMatrixFromQuaternion(linkState_wrist3[5]), (3,3))
        frame_wrist3[0:3, 3] = linkState_wrist3[4]

        # run through each frame to add ray starts and ends
        #TODO: Placement part (wo genau sollen die rays platziert werden)
        #TODO: Aufteilung vom Kuchen
        for angle in np.linspace(-np.pi, 3*np.pi/4, 8):
            #change for axis positioning of rays (im vektor) letzte Stelle 1 = Position, letzte Stelle 0 = richtung
            rays_starts.append(np.matmul(frame_wrist3, np.array([0.0, 0.05, 0.0, 1]).T)[0:3].tolist())
            rays_ends.append(np.matmul(frame_wrist3, np.array([self.ray_end * np.sin(angle), 0.05, self.ray_end * np.cos(angle), 1]).T)[0:3].tolist())


        #needs to parameters, rays_starts = 3er Tuple start position, ray ende = List of 3er Tuples 
        # for every ray that gets casted 3 start and 3 end tuples are needed in the list
        results = pyb.rayTestBatch(rays_starts, rays_ends)
        
        if self.render:
            hitRayColor = [0, 1, 0]
            missRayColor = [1, 0, 0]

            pyb.removeAllUserDebugItems()  # this will kill workspace borders if they are displayed 

            for index, result in enumerate(results):
                #same as if result[2] = 1
                if result[0] == -1:
                    pyb.addUserDebugLine(rays_starts[index], rays_ends[index], missRayColor)
                else:
                    pyb.addUserDebugLine(rays_starts[index], rays_ends[index], hitRayColor)
        # returns int between 0-1. if * length then it is the place of contact with object. If object was not touched it returns 1 
        return np.array(results, dtype=object)[:,2]  # keeps only the distance information

    def _process_raw_lidar(self, raw_lidar_data):
        raw_bucket_size = 1 / self.indicator_buckets  # 1 is the range of pybullet lidar data (from 0 to 1)
        indicator_label_diff = 2 / self.indicator_buckets  # 2 is the range of the indicator data (from -1 to 1)
        
        # lambda function to convert to indicator based on bucket size
        raw_to_indicator = lambda x : 1 if x >= 0.99 else round((np.max([(np.ceil(x/raw_bucket_size)-1),0]) * indicator_label_diff - 1),5)
        # short explanation: takes a number between 0 and 1, assigns it a bucket in the range, and returns the corresponding bucket in the range of -1 and 1
        # the round is thrown in there to prevent weird numeric appendages that came up in testing, e.g. 0.200000000004, -0.199999999999 or the like

       
        lidar_shape = 8  #number of rays
        indicator = np.zeros(lidar_shape)
        distances = np.zeros(lidar_shape)

        # conversion

        # wrist 3
        for i in range(8):
            indicator[i] = raw_to_indicator(raw_lidar_data[i])
            distances[i] = raw_lidar_data[i] * (self.ray_end - self.ray_start) + self.ray_start
    
        return indicator, distances