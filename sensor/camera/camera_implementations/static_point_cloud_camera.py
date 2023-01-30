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
    def __init__(self, robot: UR5, position: List, target: List = None, camera_args: CameraArgs = None,
                 name: str = 'default_floating', objects_to_remove: List = None,
                 **kwargs):
        super().__init__(target=target, camera_args=camera_args, name=name, **kwargs)
        self.robot = robot
        self.pos = position

        # make sure the image type is ds for depth segmentation
        assert camera_args["type"] == "ds"

        # transformation matrix for transforming pixel coordinates to real world ones
        self.tran_pix_world: np.array

        # pybullet objects to remove from point cloud
        self.objects_to_remove = objects_to_remove

        # points
        self.points: np.array

    def _adapt_to_environment(self):
        self.target = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4]
        super()._adapt_to_environment()

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            self._adapt_to_environment()
            image = self._get_image()
            self.points = self._depth_img_to_point_cloud(image[:, :, 1])
            self.points = self._prepreprocess_point_cloud(self.points, image[:, :, 1])
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def get_observation(self):
        return {self.output_name: self.points}

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
        # Points that have the same color as the first point in the point cloud are removed
        # Points that have the color [60, 180, 75] are removed, as this is the color used for the target point
        segImg = segImg.flatten()
        select_mask = np.logical_not(np.isin(segImg, self.objects_to_remove))
        points = points[select_mask]

        # pyb.removeAllUserDebugItems()
        # pyb.addUserDebugPoints(points, np.tile([255, 0, 0], points.shape[0]).reshape(points.shape))
        # sleep(25325)
        return points.astype(np.float32)
