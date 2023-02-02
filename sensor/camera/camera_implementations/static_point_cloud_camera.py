import numpy as np
import pybullet as pyb
from typing import Union, List, Dict, TypedDict
from robot.robot_implementations.ur5 import UR5
from ..camera_utils import *
from ..camera import CameraBase, \
    CameraArgs  # to prevent circular imports the things within the package have to be imported using the relative path
from numpy import newaxis as na
from time import time
from time import sleep
import pandas as pd
import torch


__all__ = [
    'StaticPointCloudCamera'
]


class StaticPointCloudCamera(CameraBase):
    def __init__(self, sensor_config):
        # whether to use GPU or not
        self.use_gpu = sensor_config["use_gpu"]

        super().__init__(sensor_config)

        self.robot = sensor_config["robot"]
        self.pos = sensor_config["position"]

        # transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world: np.array

        # pybullet objects to remove from point cloud
        self.objects_to_remove = sensor_config["objects_to_remove"]

        if self.use_gpu:
            self.objects_to_remove = torch.asarray(self.objects_to_remove).to("cuda:0")
        # points
        self.points: np.array = None

        # segmentation image
        self.segImg: np.array = None

        # target
        self.target = sensor_config["target"]

        # whether to update the matrices or not
        self.update_matrices = True
        if sensor_config["debug"] is None:
            self.update_matrices = False

        # image width and height
        self.width = self.camera_args['width']
        self.height = self.camera_args['height']
        self.img_resolution = self.width * self.height

        # set camera matrices once
        self._set_camera()

        # create the arrays used when taking the images and when creating the pcr to make code faster
        #self.image = np.empty((self.img_resolution, 2), dtype=np.float32)
        self.depth = np.empty(self.img_resolution, dtype=np.float32)
        self.seg_img_full = np.empty(self.img_resolution, dtype=int)
        self.W = np.arange(0, self.width)
        self.H = np.arange(0, self.height)
        self.X = ((2 * self.W - self.width) / self.width)[na, :].repeat(self.height, axis=0).flatten()
        self.Y = (-1 * (2 * self.H - self.height) / self.height)[:, na].repeat(self.width, axis=1).flatten()
        self.PixPos = np.empty((self.img_resolution, 4), dtype=np.float32)
        self.PixPos[:, 0] = self.X
        self.PixPos[:, 1] = self.Y
        self.PixPos[:, 3] = np.ones(self.img_resolution)

        if self.use_gpu:
            self.PixPos = torch.from_numpy(self.PixPos).to("cuda:0")
            self.depth = torch.empty(self.img_resolution, dtype=torch.float32).to("cuda:0")
            self.seg_img_full = torch.empty(self.img_resolution, dtype=torch.int).to("cuda:0")

    def _set_camera(self):
        if self.debug.get('position', False) or self.debug.get('orientation', False) or self.debug.get('target', False) or self.debug.get('lines', False):
            self._use_debug_params()

        self.viewMatrix = pyb.computeViewMatrix(
            cameraTargetPosition=self.target,
            cameraEyePosition= self.pos,
            cameraUpVector= self.camera_args['up_vector'],
            )

        self.projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov= self.camera_args['fov'],
            aspect=self.camera_args['aspect'],
            nearVal= self.camera_args['near_val'],
            farVal= self.camera_args['far_val'],
            )

        # set transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.projectionMatrix, dtype=np.float32).reshape([4, 4], order='F'),
                                                      np.asarray(self.viewMatrix, dtype=np.float32).reshape([4, 4], order='F')))

        if self.use_gpu:
            self.tran_pix_world = torch.from_numpy(self.tran_pix_world).to("cuda:0")

    def _get_image(self):
        # getting image
        _, _, _, depth, seg = pyb.getCameraImage(
            width=self.camera_args['width'],
            height=self.camera_args['height'],
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)

        if self.use_gpu:
            self.depth[:] = torch.asarray(depth)
            self.seg_img_full[:] = torch.asarray(seg)
        else:
            self.depth[:] = depth
            self.seg_img_full[:] = seg

        return self.depth, self.seg_img_full

    def _adapt_to_environment(self):
        pass

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            # create point cloud
            self.depth, self.seg_img_full = self._get_image()
            self.points = self._depth_img_to_point_cloud(self.depth)
            self.points, self.segImg = self._prepreprocess_point_cloud(self.points, self.seg_img_full)
            self.obstacle_cuboids = self._pcr_to_cuboid(self.points, self.segImg)
        self.cpu_time = time() - self.cpu_epoch
        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        # create point cloud
        self.depth, self.seg_img_full = self._get_image()
        self.points = self._depth_img_to_point_cloud(self.depth)
        self.points, self.segImg = self._prepreprocess_point_cloud(self.points, self.seg_img_full)
        self.obstacle_cuboids = self._pcr_to_cuboid(self.points, self.segImg)
        self.cpu_time = time() - self.cpu_epoch
        return {"point_cloud": self.points}

    def get_observation(self):
        """Point Cloud should not be added to observation space from here"""
        pass

    def _normalize(self):
        """
        don't know a good way to normalize this yet
        """
        # TODO: implement normalization
        pass

    def get_data_for_logging(self) -> dict:
        return {"pcr_sensor_update_time": self.cpu_time}


    ### point cloud methods ###
    def _depth_img_to_point_cloud(self, depth: np.array) -> np.array:
        """
        Compute a point cloud from a given depth image. The computation is done according to this stackoverflow post:
        https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        :param depth: input depth image;
        the amount of points in the image should equal the product of the camera sensors pixel width and height
        :type depth: np.array
        :return: The point cloud in the shape [width x height, 3]
        :rtype: np.array
        """
        # set depth values
        if self.use_gpu:
            self.PixPos[:, 2] = torch.flatten(2 * depth - 1)
            self.PixPos[:, 2] = (2 * depth - 1).flatten()
            points = torch.tensordot(self.PixPos, self.tran_pix_world, dims=[[1], [1]])
            points = (points / points[:, 3][:, na])[:, 0:3]
        else:
            self.PixPos[:, 2] = (2 * depth - 1).flatten()
            points = np.tensordot(self.tran_pix_world, self.PixPos, axes=(1, 1)).swapaxes(0, 1)
            points = (points / points[:, 3][:, na])[:, 0:3]
        return points

    def _prepreprocess_point_cloud(self, points: np.array, segImg: np.array) -> np.array:
        """
        Remove the points that correspond to the pybullet object specified in self.objects_to_remove
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: np.array
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: np.array
        :return: the points of the point cloud with the points for the background, robot arm and target removed
        :rtype: np.array
        """
        if self.use_gpu:
            segImg = torch.flatten(segImg)
            if self.objects_to_remove is not None:
                select_mask = torch.logical_not(torch.isin(segImg, self.objects_to_remove))
                points = points[select_mask]
                segImg = segImg[select_mask]
        else:
            if self.objects_to_remove is not None:
                select_mask = np.logical_not(np.isin(segImg, self.objects_to_remove))
                points = points[select_mask]
                segImg = segImg[select_mask]

        return points, segImg

    def _pcr_to_cuboid(self, points, segImg):
        """
        Transform point cloud into a set of cuboids for each obstacle. A cuboid consists of middle point, length,
        height and depth.
        """
        if self.use_gpu:
            objects = torch.unique(segImg)
            min_values = []
            max_values = []
            for object in objects:
                min_values.append(torch.min(points[segImg == object], dim=0)[0])
                max_values.append(torch.max(points[segImg == object], dim=0)[0])

            min_values = torch.reshape(torch.cat(min_values, dim=0), (len(objects), 3)).to("cuda:0")
            max_values = torch.reshape(torch.cat(max_values, dim=0), (len(objects), 3)).to("cuda:0")
            measures = max_values - min_values
            center_points = (max_values - min_values) / 2

            self.obstacle_cuboids = torch.cat([max_values[:, 0][:, None], min_values[:, 0][:, None],
                                               max_values[:, 1][:, None], min_values[:, 1][:, None],
                                               max_values[:, 2][:, None], min_values[:, 2][:, None],
                                               measures, center_points], dim=1).cpu().numpy()
            return self.obstacle_cuboids

        else:
            # create DataFrame with point coordinates and object ID
            df = pd.DataFrame({
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "object": segImg
            })

            df = df.groupby("object").agg(["max", "min"])
            df.columns = ["x_max", "x_min", "y_max", "y_min", "z_max", "z_min"]
            df["length"] = df["x_max"] - df["x_min"]
            df["depth"] = df["y_max"] - df["y_min"]
            df["height"] = df["z_max"] - df["z_min"]
            df["x_center"] = (df["x_max"] + df["x_min"]) / 2
            df["y_center"] = (df["y_max"] + df["y_min"]) / 2
            df["z_center"] = (df["z_max"] + df["z_min"]) / 2

            self.obstacle_cuboids = df.to_numpy().astype(np.float32)
        return self.obstacle_cuboids

