from modular_drl_env.world.obstacles.obstacle import Obstacle
import pybullet as pyb
import numpy as np 
from typing import Union
from .human_lib.human.man.man import Man
from .human_lib.human.human import applyMMMRotationToURDFJoint

class Human(Obstacle):
    """
    Implements a movable human as an obstacle as coded by Kolja and Kai.
    """

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, sim_step: float, thresh: float, scale: float=1):
        super().__init__(position, rotation, trajectory, 0)
        self.human = None
        self.scale = scale

        # guard against using this class with engines other than Pybullet
        if self.engine.engine_type != "Pybullet":
            raise Exception("Human obstacles cannot be used with engines other than Pybullet!")

        self.sim_step = sim_step
        self.trajectory_idx = 0

        self.hand_raise_iterator = 0
        self.hand_raise_iter_max = 100
        self.hand_raise_iter_min = 0
        self.hand_raise_direction = 1
        self.closeness_threshold = thresh

    def build(self) -> int:
        self.human = Man(0, partitioned=False, timestep=self.sim_step, scaling=self.scale, static=(len(self.trajectory)==0))
        #self.human.resetGlobalTransformation(self.position_orig, pyb.getEulerFromQuaternion(self.rotation_orig.tolist()))
        self.human.advance(self.position_orig, self.orientation_orig.tolist())
        self.internal_id = self.human.body_id
        # ultra hacky solution to make this work with the new engine abstraction, but should be fine because this human model will only ever work with pybullet
        name = "human_" + str(len(self.engine._geometry))
        self.engine._geometry[name] = self.internal_id
        self.object_id = name
        return name

    def move(self):
        if not self.trajectory:
            pass  # empty trajectory, do nothing
        elif len(self.trajectory) == 1:
            raise Exception("Human trajectories need to be either empty or at least two elements")
        else:  # looping trajectory
            # get the target from trajectory
            target_pos = self.trajectory[self.trajectory_idx]
            # get current position
            self.position = np.array(pyb.getBasePositionAndOrientation(self.internal_id)[0])
            # get the last target (see below why)
            last_target = self.trajectory[self.trajectory_idx -1] if self.trajectory_idx > 0 else self.trajectory[len(self.trajectory) - 1]

            # get rotation transform between last and current target
            direction = target_pos - last_target
            direction_norm = direction / np.linalg.norm(direction)

            rpy = [
                0, -np.arcsin(-direction_norm[2]), -np.arctan2(direction_norm[0], direction_norm[1])
            ]

            quat = np.array(pyb.getQuaternionFromEuler(rpy))

            if np.linalg.norm(self.position - target_pos) < self.closeness_threshold:
                self.trajectory_idx = (self.trajectory_idx + 1) % len(self.trajectory)
                self.human.resetGlobalTransformation([0, 0, 0], [0, 0, 0])
            last_target = self.trajectory[self.trajectory_idx -1] if self.trajectory_idx > 0 else self.trajectory[len(self.trajectory) - 1]

            # move human
            # this is weird due to the underlying human skeleton library
            # basically, this teleports the human to last_target and lets him face in quat direction
            # then, it performs a slice of forward movement a sim_step long
            # this is why we need to use last target here
            self.human.advance(last_target, quat)


    def raise_hands(self):
        d = 0.01
        applyMMMRotationToURDFJoint(self.human.body_id, 8, 0.8 * (d * self.hand_raise_iterator), -0.5 * (d * self.hand_raise_iterator), 0)
        pyb.resetJointState(self.human.body_id, 10, -0.5 * (d * self.hand_raise_iterator))
        applyMMMRotationToURDFJoint(self.human.body_id, 9, 0.8 * (d * self.hand_raise_iterator), 0.5 * (d * self.hand_raise_iterator), 0)
        pyb.resetJointState(self.human.body_id, 11, -0.5 * (d * self.hand_raise_iterator))
        if self.hand_raise_direction:
            self.hand_raise_iterator = min(self.hand_raise_iterator + 1, self.hand_raise_iter_max)
            if self.hand_raise_iterator == self.hand_raise_iter_max:
                self.hand_raise_direction = 0
        else:
            self.hand_raise_iterator = max(self.hand_raise_iterator -1, self.hand_raise_iter_min)
            if self.hand_raise_iterator == self.hand_raise_iter_min:
                self.hand_raise_direction = 1