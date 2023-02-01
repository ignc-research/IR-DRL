import pybullet as pyb
import torch
from gym.spaces import Box
import numpy as np
from numpy import newaxis as na
from sensor.sensor import Sensor
from time import time

def interpolate(a, b, n_interpolations, upper_limit, lower_limit):
    r = (b - a)
    factor = np.linspace(lower_limit, upper_limit, n_interpolations)
    a = np.repeat(a[na, :], n_interpolations, axis=0)
    return a + factor[:, na].dot(r[na, :])



class RobotSkeletonSensor(Sensor):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)
        # set robot
        self.robot = sensor_config["robot"]
        self.debug = sensor_config["debug"]

    def _set_skeleton(self):
        self.robot_id = self.robot.object_id

        robot_skeleton = []
        for i in range(pyb.getNumJoints(self.robot_id)):
            if i > 0:  # this removes the base link which is somewhere in the air
                # the center of mass of the base link (i == 1) floats in the air so we retrieve its frame link
                # instead links with an index of 4 or higher have the same coordinates for their link frame and their
                # center of mass
                if i == 1 or i >= 4:
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[4])
                else:
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[4])
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[0])

        self.robot_skeleton = np.asarray(robot_skeleton, dtype=np.float32).round(10)

        # add extra points along the arms of the robot
        self.robot_skeleton = np.concatenate([
            self.robot_skeleton,
            interpolate(self.robot_skeleton[1, :], self.robot_skeleton[2, :], 4, 0.3, 1.5),
            interpolate(self.robot_skeleton[3, :], self.robot_skeleton[4, :], 3, 0.3, 1.2),
        ], axis=0).astype(np.float32)

        # display skeleton points
        pyb.removeAllUserDebugItems()
        if self.debug["skeleton"]:
            for i, point in enumerate(self.robot_skeleton):
                # print(i, point[2].round(5))
                pyb.addUserDebugLine(point, point + np.array([0, 0, 0.2]), lineColorRGB=[0, 0, 255])

        # cast to tensor
        self.robot_skeleton = torch.from_numpy(self.robot_skeleton)
    def update(self, step) -> dict:
        if step % self.update_steps == 0:
            self._set_skeleton()
        return {"robot_skeleton": self.robot_skeleton}

    def reset(self):
        self.cpu_epoch = time()
        self._set_skeleton()
        self.cpu_time = time() - self.cpu_epoch
        return {"robot_skeleton": self.robot_skeleton}

    def get_observation(self) -> dict:
        """
        Returns the data currently stored. Does not perform an update.
        This must return the data in the same format as defined below in the gym space.
        """
        return {"robot_skeleton": self.robot_skeleton}

    def _normalize(self) -> dict:
        """
        don't know a good way to normalize this yet
        """
        # TODO: implement normalization
        pass

    def get_observation_space_element(self) -> dict:
        """
        Returns a dict with gym spaces to be used as a part of the observation space of the parent gym env. Called once at the init of the gym env.
        Dict keys should contain a sensible name for the data and the name of the robot, values should be associated gym spaces.
        Example for position sensor: "position_ur5_1": gym.space.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        """
        if self.add_to_observation_space:
            obs_sp_ele = dict()
            obs_sp_ele["robot_skeleton"] = Box(
                low=np.repeat(np.array([-1, -1, 1], dtype=np.float32)[na, :], 16, axis=0),
                high=np.repeat(np.array([1, 1, 2], dtype=np.float32)[na, :], 16, axis=0),
                shape=(16, 3), dtype=np.float32)
            return obs_sp_ele
        else:
            return {}
