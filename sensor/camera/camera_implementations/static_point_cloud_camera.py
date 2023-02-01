import numpy as np
import pybullet as pyb
from typing import Union, List, Dict, TypedDict, Tuple

import torch

from robot.robot_implementations.ur5 import UR5
from ..camera_utils import *
from ..camera import CameraBase, \
    CameraArgs  # to prevent circular imports the things within the package have to be imported using the relative path
from numpy import newaxis as na
from time import time

__all__ = [
    'StaticPointCloudCamera'
]


class StaticPointCloudCamera(CameraBase):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)

        self.robot = sensor_config["robot"]
        self.pos = sensor_config["position"]

        # transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world: np.array

        # pybullet objects to remove from point cloud
        self.objects_to_remove = sensor_config["objects_to_remove"]

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
        self.image = np.empty((self.img_resolution, 2), dtype=np.float32)
        self.W = np.arange(0, self.width)
        self.H = np.arange(0, self.height)
        self.X = ((2 * self.W - self.width) / self.width)[na, :].repeat(self.height, axis=0).flatten()
        self.Y = (-1 * (2 * self.H - self.height) / self.height)[:, na].repeat(self.width, axis=1).flatten()
        self.PixPos = np.empty((self.img_resolution, 4), dtype=np.float32)
        self.PixPos[:, 0] = self.X
        self.PixPos[:, 1] = self.Y
        self.PixPos[:, 3] = np.ones(self.img_resolution)

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
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.projectionMatrix).reshape([4, 4], order='F'),
                                                      np.asarray(self.viewMatrix).reshape([4, 4], order='F')))

    def _get_image(self):
        # getting image
        _, _, _, depth, seg = pyb.getCameraImage(
            width=self.camera_args['width'],
            height=self.camera_args['height'],
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)

        self.image[:, 0] = depth
        self.image[:, 1] = seg

        return self.image
    def _adapt_to_environment(self):
        pass

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            # create point cloud
            image = self._get_image()
            self.points = self._depth_img_to_point_cloud(image[:, 0])
            self.points, self.segImg = self._prepreprocess_point_cloud(self.points, image[:, 1])
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        # create point cloud
        image = self._get_image()
        self.points = self._depth_img_to_point_cloud(image[:, 0])
        self.points, self.segImg = self._prepreprocess_point_cloud(self.points, image[:, 1])
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
        self.PixPos[:, 2] = (2 * depth - 1).flatten()
        points = np.tensordot(self.tran_pix_world, self.PixPos, axes=(1, 1)).swapaxes(0, 1).astype(np.float32)
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
        segImg = segImg.flatten()
        if self.objects_to_remove is not None:
            select_mask = np.logical_not(np.isin(segImg, self.objects_to_remove))
            points = points[select_mask]
            segImg = segImg[select_mask]

        return torch.from_numpy(points), torch.from_numpy(segImg)




# torch implementation; doesnt really work yet
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

class StaticPointCloudCameraTorch(CameraBase):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)

        self.robot = sensor_config["robot"]
        self.pos = sensor_config["position"]

        # transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world: torch.Tensor

        # pybullet objects to remove from point cloud
        self.objects_to_remove = torch.asarray(sensor_config["objects_to_remove"])

        # points
        self.points: torch.Tensor

        # segmentation image
        self.segImg: torch.Tensor

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

        # # create the arrays used when taking the images and when creating the pcr to make code faster
        self.image = torch.empty((self.img_resolution, 2), dtype=torch.float32)
        self.W = torch.arange(0, self.width)
        self.H = torch.arange(0, self.height)
        self.X = torch.flatten(torch.repeat_interleave(((2 * self.W - self.width) / self.width)[None, :], self.height, dim=0))
        self.Y = torch.flatten(torch.repeat_interleave((-1 * (2 * self.H - self.height) / self.height)[None, :], self.width, dim=0))
        self.PixPos: torch.Tensor = torch.empty((self.img_resolution, 4), dtype=torch.float32)
        self.PixPos[:, 0] = self.X
        self.PixPos[:, 1] = self.Y
        self.PixPos[:, 3] = torch.ones(self.img_resolution)

    def _set_camera(self):
        if self.debug.get('position', False) or self.debug.get('orientation', False) or self.debug.get('target', False) or self.debug.get('lines', False):
            self._use_debug_params()

        self.viewMatrix = pyb.computeViewMatrix(
            cameraTargetPosition=self.target,
            cameraEyePosition=self.pos,
            cameraUpVector=self.camera_args['up_vector'],
        )

        self.projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov=self.camera_args['fov'],
            aspect=self.camera_args['aspect'],
            nearVal=self.camera_args['near_val'],
            farVal=self.camera_args['far_val'],
        )

        # set transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world = torch.linalg.inv(
            torch.matmul(reshape_fortran(torch.asarray(self.projectionMatrix), [4, 4]),
                         reshape_fortran(torch.asarray(self.viewMatrix), [4, 4])))

    def _get_image(self):
        # getting image
        _, _, _, depth, seg = pyb.getCameraImage(
            width=self.camera_args['width'],
            height=self.camera_args['height'],
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)
        self.image[:, 0] = torch.asarray(depth, dtype=torch.float32)
        self.image[:, 1] = torch.asarray(seg)

        return self.image

    def _adapt_to_environment(self):
        pass

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            # create point cloud
            image = self._get_image()
            self.points = self._depth_img_to_point_cloud(image[:, 0])
            self.points, self.segImg = self._prepreprocess_point_cloud(self.points, image[:, 1])
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        # create point cloud
        image = self._get_image()
        self.points = self._depth_img_to_point_cloud(image[:, 0])
        self.points, self.segImg = self._prepreprocess_point_cloud(self.points, image[:, 1])
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
    def _depth_img_to_point_cloud(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute a point cloud from a given depth image. The computation is done according to this stackoverflow post:
        https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        :param depth: input depth image;
        the amount of points in the image should equal the product of the camera sensors pixel width and height
        :type depth: torch.Tensor
        :return: The point cloud in the shape [width x height, 3]
        :rtype: torch.Tensor
        """
        # set depth values
        self.PixPos[:, 2] = torch.flatten(2 * depth - 1)
        points = torch.tensordot(self.PixPos, self.tran_pix_world, dims=([1], [1]))
        print(points)
        points = (points / points[:, 3][:, None])[:, 0:3]

        return points

    def _prepreprocess_point_cloud(self, points: torch.Tensor, segImg: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Remove the points that correspond to the pybullet object specified in self.objects_to_remove
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: torch.Tensor
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: torch.Tensor
        :return: the points of the point cloud and the segmentation mask
        with the points for the background, robot arm and target removed
        :rtype: Tuple[torch.Tensor]
        """
        segImg = torch.flatten(segImg)
        if self.objects_to_remove is not None:
            select_mask = torch.logical_not(torch.isin(segImg, self.objects_to_remove))
            points = points[select_mask, :]
            segImg = segImg[select_mask]

        return torch.asarray(points, dtype=torch.float32), torch.asarray(segImg, dtype=torch.int8)
