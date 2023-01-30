import pybullet as pyb
from gym.spaces import Box
import numpy as np
from numpy import newaxis as na
from sensor.sensor import Sensor
from time import time


class RobotSkeletonSensor(Sensor):
    def __init__(self, sensor_config):
        super().__init__(sensor_config)
        # set robot
        self.robot = sensor_config["robot"]

    def update(self, step) -> dict:
        if step % self.update_steps == 0:
            self.robot_id = self.robot.object_id

            robot_skeleton = []
            for i in range(pyb.getNumJoints(self.robot_id)):
                if i > 2:
                    if i == 3:
                        robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[0])
                        robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[4])
                    else:
                        robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[0])
            self.robot_skeleton = np.asarray(robot_skeleton, dtype=np.float32).round(10)
            # add 3 additional points along the arm
            self.robot_skeleton = np.append(self.robot_skeleton,
                                            ((self.robot_skeleton[1] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
            self.robot_skeleton = np.append(self.robot_skeleton,
                                            ((self.robot_skeleton[2] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
            self.robot_skeleton = np.append(self.robot_skeleton,
                                            ((self.robot_skeleton[6] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
            # add an additional point on the right side of the head
            self.robot_skeleton = np.append(self.robot_skeleton,
                                            (self.robot_skeleton[3] - 1.5 * (
                                                        self.robot_skeleton[3] - self.robot_skeleton[2]))[na, :], axis=0)
        return {"robot_skeleton": self.robot_skeleton}

    def reset(self):
        self.cpu_epoch = time()
        self.robot_id = self.robot.object_id

        robot_skeleton = []
        for i in range(pyb.getNumJoints(self.robot_id)):
            if i > 2:
                if i == 3:
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[0])
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[4])
                else:
                    robot_skeleton.append(pyb.getLinkState(self.robot_id, i)[0])
        self.robot_skeleton = np.asarray(robot_skeleton, dtype=np.float32).round(10)
        # add 3 additional points along the arm
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[1] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[2] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[6] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        # add an additional point on the right side of the head
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        (self.robot_skeleton[3] - 1.5 * (
                                                self.robot_skeleton[3] - self.robot_skeleton[2]))[na, :], axis=0)
        self.cpu_time = time() - self.cpu_epoch

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
        pass

    def get_observation_space_element(self) -> dict:
        """
        Returns a dict with gym spaces to be used as a part of the observation space of the parent gym env. Called once at the init of the gym env.
        Dict keys should contain a sensible name for the data and the name of the robot, values should be associated gym spaces.
        Example for position sensor: "position_ur5_1": gym.space.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        """
        if self.add_to_observation_space:
            obs_sp_ele = dict()
            obs_sp_ele[self.robot.name + "_skeleton"] = Box(low=-5, high=5, shape=(10, 3))
            return obs_sp_ele
        else:
            return {}
