import pybullet as pyb
from typing import Union, List, Dict, TypedDict
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

        # make sure the image type is ds for depth segmentation
        assert sensor_config["camera_args"]["type"] == "ds"

        # transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world: np.array

        # pybullet objects to remove from point cloud
        self.objects_to_remove = sensor_config["objects_to_remove"]

        # points
        self.points: np.array

        # target
        self.target = sensor_config["target"]

        # whether to update the matrices or not
        self.update_matrices = True
        if sensor_config["debug"] is None:
            self.update_matrices = False

    def _set_camera(self):
        if self.debug.get('position', False) or self.debug.get('orientation', False) or self.debug.get('target', False) or self.debug.get('lines', False):
            self._use_debug_params()

        viewMatrix = pyb.computeViewMatrix(
            cameraTargetPosition=self.target,
            cameraEyePosition= self.pos,
            cameraUpVector= self.camera_args['up_vector'],
            )

        projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov= self.camera_args['fov'],
            aspect=self.camera_args['aspect'],
            nearVal= self.camera_args['near_val'],
            farVal= self.camera_args['far_val'],
            )

        # set transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(projectionMatrix).reshape([4, 4], order='F'),
                                                      np.asarray(viewMatrix).reshape([4, 4], order='F')))

        def _set_camera_inner(): # TODO
            _, _, _, depth, seg = pyb.getCameraImage(
                width= self.camera_args['width'],
                height= self.camera_args['height'],
                viewMatrix= viewMatrix,
                projectionMatrix= projectionMatrix)

            depth, seg = np.array(depth), np.array(seg) # for compatibility with older python versions
            image = np.stack([depth, seg], axis=1)

            return image

        self.camera_ready = True
        return _set_camera_inner

    def _adapt_to_environment(self):
        pass

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            if self.update_matrices:
                self._adapt_to_environment()
            image = self._get_image()
            self.points = self._depth_img_to_point_cloud(image[:, 0])
            self.points = self._prepreprocess_point_cloud(self.points, image[:, 1])
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def get_observation(self):
        """Point Cloud should not be added to observation space from here"""
        pass

    def _normalize(self):
        """
        don't know a good way to normalize this yet
        """
        pass

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
        width = self.camera_args["width"]
        height = self.camera_args["height"]
        # set width and height
        W = np.arange(0, width)
        H = np.arange(0, height)

        # compute pixel coordinates
        X = ((2 * W - width) / width)[na, :].repeat(height, axis=0).flatten()[:, na]
        Y = (-1 * (2 * H - height) / height)[:, na].repeat(width, axis=1).flatten()[:, na]
        Z = (2 * depth - 1).flatten()[:, na]

        # transform pixel coordinates into real world ones
        num_of_pixels = width * height
        PixPos = np.concatenate([X, Y, Z, np.ones(num_of_pixels)[:, na]], axis=1)
        points = np.tensordot(self.tran_pix_world, PixPos, axes=(1, 1)).swapaxes(0, 1)
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

        # pyb.removeAllUserDebugItems()
        # pyb.addUserDebugPoints(points, np.tile([255, 0, 0], points.shape[0]).reshape(points.shape))
        # from time import sleep
        # sleep(2)
        return points.astype(np.float32)
