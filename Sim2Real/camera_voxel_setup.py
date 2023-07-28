## This script is for setting up the camera environment. It only visualises the pointcloud and voxilization part without needing an actual robot connected.

# Serial numbers for lab cameras: 
# cam_1 (table cam): 028522073665
# cam_2 (additional cam): 141322251391

#roslaunch realsense2_camera rs_camera.launch camera:=cam_1 serial_no:=028522073665 filters:=pointcloud
#roslaunch realsense2_camera rs_camera.launch camera:=cam_2 serial_no:=141322251391 filters:=pointcloud


import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import numpy as np
from time import process_time, time
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2
import sys
from modular_drl_env.gym_env.environment import ModularDRLEnv
from modular_drl_env.util.configparser import parse_config
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
from modular_drl_env.world.obstacles.shapes import Box, Sphere
from modular_drl_env.planner.planner_implementations.rrt import BiRRT
import ros_numpy
from stable_baselines3 import PPO
import open3d as o3d
from time import sleep
import yaml
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

import pandas as pd

import torch
from collections import deque
from scipy.spatial import cKDTree

from collections import deque

import pybullet as pyb

import pickle

from voxelization import get_voxel_cluster, set_clusters, get_neighbouring_voxels_idx, statistical_outlier_removal



JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class listener_node_one:
    def __init__(self, num_voxels, point_cloud_static):
        # yaml config file
        self.config_path = '/home/moga/Desktop/IR-DRL/Sim2Real/config_data/config.yaml'
        self.config = self.load_config(self.config_path)

        # robot data
        self.end_effector_link_id = 6
        self.end_effector_xyz = None
        self.end_effector_rpy = None
        self.end_effector_quat = None
        self.end_effector_xyz_sim = None

        self.joints = None
        self.velocities = None

        self.running_inference = False  # flag for interference between DRL and symsinc to prevent both from running parallel
        self.point_cloud_static = point_cloud_static
        self.static_done = False
        self.num_voxels = num_voxels
        self.color_voxels = True # einfärben der Voxels
        self.control_mode = True   # True=pos, False=joints
        self.startup = True
        self.drl_success = False
        self.dist_threshold = self.config['dist_threshold']

        self.inference_steps = 0
        self.inference_done = False
        self.camera_calibration = False
        self.dont_voxelize = False
        self.use_gpu = self.config['use_gpu'] # torch.cuda.is_available()
        self.use_sor = self.config['use_sor']
        self.is_moving_voxels = False       

        # cbGetPointcloud
        # storage attributes for all the camera data
        self.num_cameras = 2
        self.points_raw = [None for _ in range(self.num_cameras)]
        self.colors = [None for _ in range(self.num_cameras)]
        self.data_set_guard = [False for _ in range(self.num_cameras)]

        # cbPointcloudToPybullet
        self.camera_transform_to_pyb_origin = [np.eye(4) for _ in range(self.num_cameras)]
        self.pyb_to_camera = [np.eye(4) for _ in range(self.num_cameras)]
        cam_config = self.config['camera_transform_to_pyb_origin']
        config_matrix = [(R.from_euler('xyz', np.array(ele["rpy"]), degrees = True)).as_matrix() for ele in cam_config]
        for i in range(self.num_cameras):
            self.camera_transform_to_pyb_origin[i][:3, :3] = config_matrix[i]
            self.camera_transform_to_pyb_origin[i][:3, 3] = np.array(cam_config[i]['xyz'])
            self.pyb_to_camera[i][:3, :3] = config_matrix[i].T
            self.pyb_to_camera[i][:3, 3] = np.matmul(-(config_matrix[i].T), np.array(cam_config[i]['xyz']))

        # pre-allocate the rotation matrix on the gpu in case we use it
        if self.use_gpu:
            self.camera_transform_gpu = [torch.from_numpy(matrix).to('cuda') for matrix in self.camera_transform_to_pyb_origin]
        
        # boundaries for the point cloud, such that we constrain it into a statically determined box
        # useful for normalized clustering later on
        self.points_lower_bound = np.array([-0.5, -1, -0.1, 1])
        self.points_upper_bound = np.array([1.2, 1.25, 1.5, 1])       

        # we also rotate the bounds into camera space, such that we can apply them for filtering before rotating the raw point cloud
        self.points_lower_bound_camera = [None for _ in range(self.num_cameras)]
        self.points_upper_bound_camera = [None for _ in range(self.num_cameras)]
        for i in range(self.num_cameras):
            tmp = np.matmul(self.pyb_to_camera[i], self.points_lower_bound)
            tmp1 = np.matmul(self.pyb_to_camera[i], self.points_upper_bound)
            self.points_lower_bound_camera[i] = np.min([tmp, tmp1], axis=0)
            self.points_upper_bound_camera[i] = np.max([tmp, tmp1], axis=0)
            if self.use_gpu:
                self.points_lower_bound_camera[i] = torch.from_numpy(self.points_lower_bound_camera[i]).to('cuda')
                self.points_upper_bound_camera[i] = torch.from_numpy(self.points_upper_bound_camera[i]).to('cuda')

        # voxelization attributes
        self.voxel_size = self.config['voxel_size']
        self.robot_voxel_safe_distance = self.config['robot_voxel_safe_distance']
        
        #change to 5 if you want more safety with dynamic obstacles, but also more delay
        self.inference_steps_per_pointcloud_update = 1
        
        
        #part for voxel clustering
        self.enable_clustering = self.config['enable_clustering']#enable or disable clustering for performance reasons
        self.robot_voxel_cluster_distance = 0.3 #TODO: optimize this
        self.neighbourhood_threshold = np.sqrt(2)*self.voxel_size + self.voxel_size/10
        self.voxel_cluster_threshold = self.config['voxel_cluster_threshold'] #TODO: Variabel an der anzahl von voxeln ändern nicht hardcoden
        self.voxel_centers = None
        
          # custom sim2real config parsen
        #_, env_config = parse_config("/home/moga/Desktop/IR-DRL/configs/S2R/obstsensor_trajectory_PPO.yaml", False) #False = kein Training
        _, env_config = parse_config("/home/moga/Desktop/IR-DRL/configs/S2R/s2rexperiment_benno_config_voxels.yaml", False)
        env_config["env_id"] = 0
        # mit der config env starten
        self.env = ModularDRLEnv(env_config)
        self.virtual_robot = self.env.robots[0] # env initialisiert robot, wird für spätere Verwendung in der virtual_robot variable gespeichert
        #self.env.reset()  # taken out, replace by command below
        self.custom_env_reset()
        # now that the robot is there, we can get the action size
        self.actions = np.zeros((100, len(self.virtual_robot.all_joints_ids)))

        self.voxels = []        # voxel objects in pybullet simulation
        self.pos_nowhere= np.array([0,0,-100])
        # initialize probe_voxel and obstacle_voxels
        self.initialize_voxels()
       
        pyb_u.toggle_rendering(False)
        self.virtual_robot.goal.delete_visual_aux()
        pyb_u.toggle_rendering(True)

        # load DRL model
        #self.model = PPO.load("/home/moga/Desktop/IR-DRL/models/weights/model_interrupt.zip")  # trajectory
        self.model = PPO.load("/home/moga/Desktop/IR-DRL/models/weights/model_trained_voxels.zip")  # no trajectory


        #self.planner.joint_ids = [pyb_u.pybullet_joints_ids[self.virtual_robot.object_id, joint_id] for joint_id in self.virtual_robot.all_joints_ids]
        #self.planner.joint_ids_u = self.virtual_robot.all_joints_ids

        #optional: enable to see if there are differences between the env and model
        """from modular_drl_env.util.misc import analyse_obs_spaces
        analyse_obs_spaces(self.env.observation_space, self.model.observation_space)"""
        
        print("[Listener] Using GPU support for voxelization." if self.use_gpu else "[Listener] Using CPU for voxelization.")
        print("[Listener] Moving robot into resting pose")
       
        sleep(1)
        print("[Listener] Started callbacks for raw pointcloud data for default camera.")
        rospy.Subscriber("/cam_1/depth/color/points", PointCloud2, self.cbGetPointcloud_0)
        if self.num_cameras == 2:
            print("[Listener] Started callbacks for raw pointcloud data for second camera.")
            rospy.Subscriber("/cam_2/depth/color/points", PointCloud2, self.cbGetPointcloud_1)
        if self.num_cameras == 3:
            print("[Listener] Started callbacks for raw pointcloud data for third camera.")
            rospy.Subscriber("/cam_3/depth/color/points", PointCloud2, self.cbGetPointcloud_2)
        print("[Listener] Started callback for pointcloud voxelization.")

        self.start_time = time() #for recording voxel purposes
        rospy.Timer(rospy.Duration(secs=1/240), self.cbPointcloudToPybullet)    
        sleep(1)
        print("[Listener] Started callback for controlling robot(needed for internal program logic even if not controlling the robot)")
        rospy.Timer(rospy.Duration(secs=1/120), self.cbCalibration)

        print("[Listener] initialized node")
        rospy.spin() #Lässt rospy immer weiter laufen

    def cbCalibration(self, event):
        
        inp = input("[cbCalibrate] Enter (c) to calibrate camera position or\n[cbCalibrate] (v) to adjust voxel size: \n")
        if inp[0] == "c": # calibrates the camera
            self._control_calibrate()
        if inp[0] == "v": # voxelsize
            self._control_voxelsettings()

    def _control_calibrate(self):
        inp = input("[cbCalibrate] Choose camera number (between 0 and "+ str(self.num_cameras-1) +"): \n")
        i = int(inp)
        was_static = self.point_cloud_static
        self.point_cloud_static = False
        self.camera_calibration = True                   
        print("[cbControl] current (virtual) camera position: " + str(self.camera_transform_to_pyb_origin[i][:3, 3]))
        print("[cbControl] current (virtual) camera rpy: " + str(self.config['camera_transform_to_pyb_origin'][i]['rpy']))  
        inp = input("[cbControl] Enter a camera position: \n")
        inp = inp.split(" ")
        inp2 = input("[cbControl] Enter a camera rotation in extrinsic XYZ Euler format and in degrees: \n")
        inp2 = inp2.split(" ")
        try:
            inp = [float(ele) for ele in inp]
            inp2 = [float(ele) for ele in inp2]
        except ValueError:
            print("[cbControl] Input in wrong format, try again!")
            self.camera_calibration = False
            return
        # Update the config dictionary
        self.config['camera_transform_to_pyb_origin'][i]['xyz'] = [float(value) for value in inp]
        self.config['camera_transform_to_pyb_origin'][i]['rpy'] = [float(value) for value in inp2]
        # Save the updated configuration to the file
        self.save_config(self.config_path, self.config)
        # Reload the configuration to get the latest values
        self.config = self.load_config(self.config_path)

        self.camera_transform_to_pyb_origin[i][:3, :3] = (R.from_euler('xyz', np.array(inp2), degrees = True)).as_matrix()
        self.camera_transform_to_pyb_origin[i][:3, 3] = np.array(inp)
        if self.use_gpu:
            self.camera_transform_gpu[i] = torch.from_numpy(self.camera_transform_to_pyb_origin[i]).to('cuda')
        self.pyb_to_camera[i][:3, :3] = self.camera_transform_to_pyb_origin[i][:3, :3].T
        self.pyb_to_camera[i][:3, 3] = np.matmul(-(self.camera_transform_to_pyb_origin[i][:3, :3].T), np.array(inp))
        tmp = np.matmul(self.pyb_to_camera[i], self.points_lower_bound)
        tmp1 = np.matmul(self.pyb_to_camera[i], self.points_upper_bound)
        self.points_lower_bound_camera[i] = np.min([tmp, tmp1], axis=0)
        self.points_upper_bound_camera[i] = np.max([tmp, tmp1], axis=0)
        if self.use_gpu:
            self.points_lower_bound_camera[i] = torch.from_numpy(self.points_lower_bound_camera[i]).to('cuda')
            self.points_upper_bound_camera[i] = torch.from_numpy(self.points_upper_bound_camera[i]).to('cuda')
        self.camera_calibration = False
        self.point_cloud_static = was_static
        self.static_done = False

    def _control_voxelsettings(self):
        self.dont_voxelize = True
        print("[cbControl] Current voxel size: " + str(self.voxel_size))
        print("[cbControl] Current robot voxel safe distance: " + str(self.robot_voxel_safe_distance))
        print("[cbControl] Current minimal cluster size: " + str(self.voxel_cluster_threshold))
        print("[cbControl] Current number of voxels: " + str(self.num_voxels))
        inp = input("[cbControl] Enter new voxel size as a float:\n")
        try:
            val = float(inp)
        except ValueError:
            print("[cbControl] Invalid value for voxel size!")
        self.voxel_size = val
        self.config['voxel_size'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter new robot voxel safe distance as a float:\n")
        try:
            val = float(inp)
        except ValueError:
            print("[cbControl] Invalid value for robot voxel safe distance!")
        self.robot_voxel_safe_distance = val
        self.config['robot_voxel_safe_distance'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter a new minimal cluster size as an int:\n")
        try: 
            val = int(inp)
        except ValueError:
            print("[cbControl] Invalid value for cluster size!")
        self.voxel_cluster_threshold = val
        self.config['voxel_cluster_threshold'] = val
        self.save_config(self.config_path, self.config)
        inp = input("[cbControl] Enter a new number of voxels as an int:\n")
        try: 
            val = int(inp)
        except ValueError:
            print("[cbControl] Invalid value for number of voxels!")
        self.num_voxels = val
        self.delete_voxels()
        self.initialize_voxels()
        self.dont_voxelize = False
    
    # for camera 1
    def cbGetPointcloud_0(self, data):
        #print("cbGetPointcloud got called.1")
        np_data = ros_numpy.numpify(data)
        points = np.ones((np_data.shape[0], 4))
        if len(points) == 0:
            print("[cbGetPointcloud] No point cloud data received, check the camera and its ROS program!")
            return
        points[:, 0] = np_data['x']
        points[:, 1] = np_data['y']
        points[:, 2] = np_data['z']
        color_floats = np_data['rgb']  # float value that compresses color data
        # convert float values into 3-vector with RGB intensities, normalized to between 0 and 1
        color_floats = np.ascontiguousarray(color_floats)
        #color_floats = (color_floats.view(dtype=np.uint8).reshape(color_floats.shape + (4,))[:,:3].astype(np.float64)) / 255
        #This code reverses the last dimension to change the order of RGB to BGR. This is necessary as the order of our colors is not rgb initially.
        color_floats = (color_floats.view(dtype=np.uint8).reshape(color_floats.shape + (4,))[:,:3][..., ::-1].astype(np.float64)) / 255
        # transfer data into class variables where other callbacks can get them
        self.data_set_guard[0] = True
        self.colors[0] = color_floats
        self.points_raw[0] = points
        self.data_set_guard[0] = False
    
    #for camera 2
    def cbGetPointcloud_1(self, data):
        np_data = ros_numpy.numpify(data)
        points = np.ones((np_data.shape[0], 4))
        if len(points) == 0:
            print("[cbGetPointcloud] No point cloud data received, check the camera and its ROS program!")
            return
        points[:, 0] = np_data['x']
        points[:, 1] = np_data['y']
        points[:, 2] = np_data['z']
        color_floats = np_data['rgb']  # float value that compresses color data
        # convert float values into 3-vector with RGB intensities, normalized to between 0 and 1
        color_floats = np.ascontiguousarray(color_floats)
        #color_floats = (color_floats.view(dtype=np.uint8).reshape(color_floats.shape + (4,))[:,:3].astype(np.float64)) / 255
        #This code reverses the last dimension to change the order of RGB to BGR. This is necessary as the order of our colors is not rgb initially.
        color_floats = (color_floats.view(dtype=np.uint8).reshape(color_floats.shape + (4,))[:,:3][..., ::-1].astype(np.float64)) / 255
        # transfer data into class variables where other callbacks can get them
        self.data_set_guard[1] = True
        self.colors[1] = color_floats
        self.points_raw[1] = points
        self.data_set_guard[1] = False

    def cbGetPointcloud_2(self, data):    
        pass

    def cbPointcloudToPybullet(self, event):
        #print("cbPointcloudToPybullet got called.2")
    # callback for PointcloudToPybullet
    # static = only one update
    # dynamic = based on update frequency        
        if self.points_raw is not None and not self.dont_voxelize: #and not self.running_inference:  # catch first time execution scheduling problems
            if self.point_cloud_static:
                if self.static_done:
                    return
                else:
                    self.static_done = True
                    self.PointcloudToVoxel()
                    self.VoxelsToPybullet()
            else:
                self.PointcloudToVoxel()
                self.VoxelsToPybullet()
          
        
    def PointcloudToVoxel(self):
       # print("PointcloudToVoxel got called 3")
        # if using GPU, points and colors are torch tensors
        # if using CPU, points and colors are numpy arrays
        not_none_points = [(ele is not None) for ele in self.points_raw]
        points = [None for _ in self.points_raw]
        colors = [None for _ in self.points_raw]
        if True in not_none_points:
            # wait for the data getter callback to finish
            for i in range(self.num_cameras):
                if not_none_points[i]:
                    while self.data_set_guard[i]:
                        sleep(0.0001)
                    points[i] = self.points_raw[i]
                    colors[i] = self.colors[i]       

            if self.use_gpu:
                for i in range(self.num_cameras):
                    if not_none_points[i]:
                        colors[i] = torch.from_numpy(colors[i]).to('cuda')
                        points[i] = torch.from_numpy(points[i]).to('cuda')

            # cut off all points that are outside of the area of interest as defined by the user
            # we use the rotated boundaries to do this on the raw point data
            # this reduces the number of points drastically before we go to the costly rotation of them all
            
            for i in range(self.num_cameras):
                if not_none_points[i]:
                    lower_mask = (points[i] >= self.points_lower_bound_camera[i]).all(axis=1)
                    upper_mask = (points[i] <= self.points_upper_bound_camera[i]).all(axis=1)
                    if self.use_gpu:
                        in_boundary_mask = torch.logical_and(lower_mask, upper_mask)
                    else:
                        in_boundary_mask = np.logical_and(lower_mask, upper_mask)
                    points[i] = points[i][in_boundary_mask]
                    colors[i] = colors[i][in_boundary_mask]

            # rotate raw points into PyBullet coordinate system
            if not self.use_gpu:
                ############## CPU #######################  
                for i in range(self.num_cameras):
                    if not_none_points[i]:
                        points[i] = np.dot(self.camera_transform_to_pyb_origin[i], points[i].T).T.reshape(-1,4)   
                points = [points[i] for i in range(self.num_cameras) if not_none_points[i]]
                colors = [colors[i] for i in range(self.num_cameras) if not_none_points[i]]
                points = np.concatenate(points, axis=0)
                colors = np.concatenate(colors, axis=0)
            else:
                #################### GPU ####################################
                # points = torch.from_numpy(points).to('cuda')
                # colors = torch.from_numpy(colors).to('cuda')
                for i in range(self.num_cameras):
                    if not_none_points[i]:
                        points[i] = torch.matmul(self.camera_transform_gpu[i], points[i].T).T.reshape(-1,4)
                points = [points[i] for i in range(self.num_cameras) if not_none_points[i]]
                colors = [colors[i] for i in range(self.num_cameras) if not_none_points[i]]
                points = torch.cat(points, dim=0)
                colors = torch.cat(colors, dim=0)

            # remove homogeneous component, don't need it anymore after the rotation  
            points = points[:, :3]  

            if not self.camera_calibration:
                
                # filter objects near endeffector
                ee_pos, _, _, _ = pyb_u.get_link_state(self.virtual_robot.object_id, self.virtual_robot.end_effector_link_id)
                # TODO may need to be adjusted with KINECT Cam
                if self.use_gpu:
                    mask_offset1 = torch.from_numpy(ee_pos) + torch.tensor([0.1, -0.1, 0.2])
                    mask_offset1 = mask_offset1.to('cuda')
                    mask_offset2 = torch.from_numpy(ee_pos) + torch.tensor([0, 0, 0.15])
                    mask_offset2 = mask_offset2.to('cuda')
                else:
                    mask_offset1 = ee_pos + np.array([0.1, -0.1, 0.2])
                    mask_offset2 = ee_pos + np.array([0, 0, 0.15])
                mask_left = self.delete_points_by_circle_center(points, mask_offset1)  # TODO: maybe find a general way to do this
                mask_above = self.delete_points_by_circle_center(points, mask_offset2)
                if self.use_gpu:
                    not_close_mask = torch.logical_and(mask_left, mask_above)
                else:
                    not_close_mask = np.logical_and(mask_left, mask_above)
                # filter by bool_array
                points = points[not_close_mask] 
                colors = colors[not_close_mask]

            # safety escape: the code below this will crash in case no points survive the filtering process
            if len(points) == 0:
                return
            
            # dump the results until now from the GPU
            if self.use_gpu:
                points = points.cpu().numpy()
                colors = colors.cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            
            # convert it into open 3d format
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors) # makes voxels have averaged colors of points, colors have to be normalize to between 0 and 1

            #print("PCD",pcd)
            if self.use_sor:
                #print("Using SOR:")
                pcd = statistical_outlier_removal(pcd)
         
            
           

            #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size=self.voxel_size, min_bound=self.points_lower_bound[:3], max_bound=self.points_upper_bound[:3])

            
            voxel_data = [(voxel.grid_index, voxel.color) for voxel in voxel_grid.get_voxels()]
            voxel_centers, voxel_colors = zip(*voxel_data)
            

            voxel_centers = np.array(voxel_centers)
            voxel_colors = np.array(voxel_colors)
            # Transform Voxel centers into xyz Coordinates            
            voxel_centers = voxel_centers * self.voxel_size + self.points_lower_bound[:3]# + offset_min

            if not self.camera_calibration:
                pyb_u.toggle_rendering(False)
                # move robot to real position, in case it's somewher else due to a planner or else
                joints_now, _ = pyb_u.get_joint_states(self.virtual_robot.object_id, self.virtual_robot.all_joints_ids)
                not_delete_mask = np.zeros(shape=(voxel_centers.shape[0],), dtype=bool)
     
                for idx, point in enumerate(voxel_centers):
                    pyb_u.set_base_pos_and_ori(self.probe_voxel.object_id, point, np.array([0, 0, 0, 1]))
                    self.probe_voxel.position = point
                    #checks if point is in close distance of the robot
                    query = pyb.getClosestPoints(pyb_u.to_pb(self.probe_voxel.object_id), pyb_u.to_pb(self.virtual_robot.object_id), self.robot_voxel_safe_distance)      
                    not_delete_mask[idx] = False if query else True
                voxel_centers = voxel_centers[not_delete_mask]
                voxel_colors = voxel_colors[not_delete_mask]
                # move robot back to where it was when filtering started
                self.virtual_robot.moveto_joints(joints_now, False, self.virtual_robot.all_joints_ids)
                if not self.is_moving_voxels:
                    pyb_u.toggle_rendering(True)
            pyb_u.set_base_pos_and_ori(self.probe_voxel.object_id, self.pos_nowhere, np.array([0, 0, 0, 1])) # TODO: Find out why this causes trouble, MOVES away probe voxel
            self.probe_voxel.position = self.pos_nowhere
            

            if self.enable_clustering:
            # get voxel_clusters
                voxel_clusters = get_voxel_cluster(voxel_centers, self.neighbourhood_threshold)
                # find clusters below a cluster size
                cluster_numbers, counts = np.unique(voxel_clusters, return_counts=True)
           
                include_cluster = cluster_numbers[np.invert(counts < self.voxel_cluster_threshold)]
                # remove voxels belonging to small clusters
                include_cluster_idx = np.isin(voxel_clusters, include_cluster)
                voxel_centers = voxel_centers[include_cluster_idx] 
                voxel_colors = voxel_colors[include_cluster_idx]
                

            self.voxel_centers = voxel_centers
            self.voxel_colors = voxel_colors

    def VoxelsToPybullet(self):
        #print("VoxelsToPybullet got called. 4")
        if self.voxel_centers is not None:
            pyb_u.toggle_rendering(False)
            self.is_moving_voxels = True
            # update voxel positions
            for idx, voxel in enumerate(self.voxels):
                if idx >= len(self.voxel_centers):
                    # set all remaining voxels to nowhere
                    for i in range(idx, len(self.voxels)):
                        self.voxels[i].position = self.pos_nowhere
                        pyb_u.set_base_pos_and_ori(self.voxels[i].object_id, self.pos_nowhere, np.array([0, 0, 0, 1]))
                    break
                pyb_u.set_base_pos_and_ori(voxel.object_id, self.voxel_centers[idx], np.array([0, 0, 0, 1]))          
                voxel.position = self.voxel_centers[idx]
            # calculate new colors
            if self.color_voxels and len(self.voxel_centers) != 0:
                # voxel_norms = np.linalg.norm(self.voxel_centers - self.camera_transform_to_pyb_origin[:3, 3], axis=1)
                # max_norm, min_norm = np.max(voxel_norms), np.min(voxel_norms)
                # voxel_norms = (voxel_norms - min_norm) / (max_norm -  min_norm)
                # colors = np.ones((len(voxel_norms), 4))
                # colors = np.multiply(colors, voxel_norms.reshape(len(voxel_norms),1))
                # colors[:,3] = 1
                # colors[:,2] *= 0.5
                # colors[:,1] *= 0.333
                # colors[:,:3] = 1 - colors[:, :3]
                ones = np.ones((self.voxel_colors.shape[0], 1), dtype=np.float32)
                new_colors = np.concatenate([self.voxel_colors, ones], axis=1)
                for idx, voxel in enumerate(self.voxels):
                    if idx >= len(self.voxel_centers):
                        break
                    pyb.changeVisualShape(pyb_u.to_pb(voxel.object_id), -1, rgbaColor=new_colors[idx])
            self.is_moving_voxels = False    
            pyb_u.toggle_rendering(True)
            
    
    def delete_points_by_circle_center(self, points, pos):
        #print("delete_points_by_circle_center got called. 5")
        # returns Boolean array or tensor
        if self.use_gpu:
            probe_norm = torch.norm((points - pos), dim=1)
        else:
            probe_norm = np.linalg.norm((points - pos), axis=1)
        return probe_norm > self.robot_voxel_safe_distance
    
    @staticmethod
    def pybullet_distance_check(voxel_id, qwe):
        pass

    def initialize_voxels(self):
        # generate probe_voxel and obstacle_voxels
        pyb_u.toggle_rendering(False)
        #self.probe_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 1, 1, 0])
        self.probe_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], sim_step=self.env.sim_step, sim_steps_per_env_step=self.env.sim_steps_per_env_step, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 1, 1, 0])
        self.probe_voxel.build()
        # self.goal_sphere = Sphere(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[],radius = 0.5, move_step=0, color=[1, 1, 1, 0])
        for i in range(self.num_voxels):
            #new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=np.concatenate((np.random.uniform(size=(3,)), np.ones(1))))
            new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], sim_step=self.env.sim_step, sim_steps_per_env_step=self.env.sim_steps_per_env_step, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=[1, 0, 0, 1])
            self.voxels.append(new_voxel)
            new_voxel.build()
            self.env.world.obstacle_objects.append(new_voxel)
        self.env.world.active_objects = self.env.world.obstacle_objects
        pyb_u.toggle_rendering(True)

    def delete_voxels(self):
        pyb_u.toggle_rendering(False)
        for voxel in self.voxels:
            #initialise pybullet again
            pyb.removeBody(pyb_u.to_pb(voxel.object_id))
            del voxel
        self.env.world.obstacle_objects = []
        self.env.world.active_objects = []
        self.voxels = []
        pyb_u.toggle_rendering(True)

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    
    def save_config(self, file_path, config):
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)

    def custom_env_reset(self):
        # replaces a call to env reset
        # we do this to avoid a few problems that arise when using the env.reset method outside of the context of training
        # or inference without sim2real functionality
        self.env.steps_current_episode = 0
        self.env.sim_time = 0
        self.env.cpu_time = 0
        self.env.inference_time = 0
        self.env.cpu_reset_epoch = process_time()
        self.env.reward = 0
        self.env.reward_cumulative = 0
        self.env.log = []
        pyb_u.toggle_rendering(False)
        self.env.world.reset(0)
        if not self.is_moving_voxels:
            pyb_u.toggle_rendering(True)
        self.env.active_robots = [True for _ in self.env.robots]
        for sensor in self.env.sensors:
            sensor.reset()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True, disable_signals=True) 
    listener = listener_node_one(num_voxels=5000, point_cloud_static=False)
