import numpy as np
from obstacles.base_obstacle import BaseObstacle
from .human.man.man import Man
from .human.human import applyMMMRotationToURDFJoint
import pybullet as p

DISTANCE_THRESHOLD = 4


def getDistanceToRobot(robot, end_effector_id, object_id, joint_id):
    jointState = p.getLinkState(object_id, joint_id)
    jointPosition = jointState[4]
    jointState2 = p.getLinkState(robot, end_effector_id)
    jointPosition2 = jointState2[4]

    distance = np.linalg.norm(np.array(jointPosition) - np.array(jointPosition2))

    return distance


class HumanObstacleStatic(BaseObstacle):

    def __init__(self, position, rotation, scale=1, robot_id=1) -> None:
        super().__init__(position, rotation, scale)
        self.human = Man(0, partitioned=False, scaling=scale, static=True)
        self.human.resetGlobalTransformation(position, p.getEulerFromQuaternion(rotation))
        self.i = 0
        self.robot_id = robot_id
        self.d = 0.01
        self.i_max = 100
        self.robot_is_close = False

    def step(self):
        if getDistanceToRobot(self.robot_id, 7, self.human.body_id, 12) <= DISTANCE_THRESHOLD:
            self.robot_is_close = True
        if self.robot_is_close:
            applyMMMRotationToURDFJoint(self.human.body_id, 8, 0.8 * (self.d * self.i), -0.5 * (self.d * self.i), 0)
            p.resetJointState(self.human.body_id, 10, -0.5 * (self.d * self.i))
            applyMMMRotationToURDFJoint(self.human.body_id, 9, 0.8 * (self.d * self.i), 0.5 * (self.d * self.i), 0)
            p.resetJointState(self.human.body_id, 11, -0.5 * (self.d * self.i))
            self.i = min(self.i + 1, self.i_max)
