from obstacles.base_obstacle import BaseObstacle
from .human.man.man import Man
from helpers.helpers import getPosition, getRotation
import pybullet as p
import math
import numpy

DONE_THRESHOLD = 1


class HumanObstacle(BaseObstacle):
    timestep = .01
    current_move_point = 0
    move_transforms = []

    def __init__(self, position, rotation, scale, params) -> None:
        super().__init__(position, rotation, scale)
        self.human = Man(0, partitioned=True, scaling=scale)
        self.move_transforms = list(map(lambda x: {"position": getPosition(x)}, params["move"]))

    def vec3_norm(self, vec):
        return math.sqrt((math.pow(vec[0], 2) + math.pow(vec[1], 2) + math.pow(vec[2], 2)))

    def get_distance(self, a, b):
        return self.vec3_norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]])

    def get_arr_diff(self, a, b):
        return list([(b[i] - x) for i, x in enumerate(a)])

    def is_done(self, a):
        return a[0] == 0 and a[1] == 0 and a[2] == 0

    def last_pos(self):
        if self.current_move_point > 0:
            return self.move_transforms[self.current_move_point - 1]["position"]
        else:
            return self.move_transforms[len(self.move_transforms) - 1]["position"]

    def normalize_vec3(self, vec):
        norm = self.vec3_norm(vec)
        return [vec[0] / norm, vec[1] / norm, vec[2] / norm]

    def step(self):
        target_pos = self.move_transforms[self.current_move_point]["position"]
        print(self.move_transforms)
        curr_pos = p.getBasePositionAndOrientation(self.human.body_id)[0]
        last_pos = self.last_pos()

        direction = self.get_arr_diff(last_pos, target_pos)
        direction_normalized = self.normalize_vec3(direction)

        rotation = {
            "rotation": {
                "p": 0,
                "r": -math.asin(-direction_normalized[2]) * (180 / math.pi),
                "y": -math.atan2(direction_normalized[0], direction_normalized[1]) * (180 / math.pi)
            }
        }
        quat = getRotation(rotation)

        # p.addUserDebugLine([last_pos[0],last_pos[1],last_pos[2]], [last_pos[0]+ direction_normalized[0],last_pos[1]+ direction_normalized[1],last_pos[2] + direction_normalized[2]], [1,0,0], 3)
        # p.addUserDebugLine([target_pos[0],target_pos[1],target_pos[2] - 1], [target_pos[0],target_pos[1],target_pos[2] + 1], [0,1,0], 3)

        if self.get_distance(curr_pos, target_pos) < DONE_THRESHOLD:
            self.current_move_point = (self.current_move_point + 1) % len(self.move_transforms)
            target_pos = self.move_transforms[self.current_move_point]["position"]
            self.human.resetGlobalTransformation([0, 0, 0], [0, 0, 0])

        self.human.advance(self.last_pos(), quat)
