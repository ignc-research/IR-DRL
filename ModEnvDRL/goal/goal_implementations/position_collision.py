from ModEnvDRL.goal.goal import Goal
import numpy as np
from ModEnvDRL.robot.robot import Robot
from gym.spaces import Box
import pybullet as pyb

__all__ = [
    'PositionCollisionGoal'
]

class PositionCollisionGoal(Goal):
    """
    This class implements a goal of reaching a certain position while avoiding collisions.
    The reward function follows Yifan's code.
    """

    def __init__(self, robot: Robot, 
                       normalize_rewards: bool, 
                       normalize_observations: bool,
                       train: bool,
                       add_to_logging: bool,
                       max_steps: int,
                       continue_after_success:bool, 
                       reward_success=10, 
                       reward_collision=-10,
                       reward_distance_mult=-0.01,
                       dist_threshold_start=3e-1,
                       dist_threshold_end=1e-2,
                       dist_threshold_increment_start=1e-2,
                       dist_threshold_increment_end=1e-3,
                       dist_threshold_overwrite:float=None,
                       dist_threshold_change:float=0.8):
        super().__init__(robot, normalize_rewards, normalize_observations, train, True, add_to_logging, max_steps, continue_after_success)

        # set output name for observation space
        self.output_name = "PositionGoal_" + self.robot.name

        # set the flags
        self.needs_a_position = True
        self.needs_a_rotation = False

        # set the reward that's given if the ee reaches the goal position and for collision
        self.reward_success = reward_success
        self.reward_collision = reward_collision
        
        # multiplicator for the distance reward
        self.reward_distance_mult = reward_distance_mult

        # set the distance thresholds and the increments for changing them
        self.distance_threshold = dist_threshold_start if self.train else dist_threshold_end
        if dist_threshold_overwrite:  # allows to set a different startpoint from the outside
            self.distance_threshold = dist_threshold_overwrite
        self.distance_threshold_start = dist_threshold_start
        self.distance_threshold_end = dist_threshold_end
        self.distance_threshold_increment_start = dist_threshold_increment_start
        self.distance_threshold_increment_end = dist_threshold_increment_end
        self.distance_threshold_change = dist_threshold_change

        # set up normalizing constants for faster normalizing
        #     reward
        max_reward_value = max(abs(self.reward_success), abs(self.reward_collision))
        self.normalizing_constant_a_reward = 2 / (2 * max_reward_value)
        self.normalizing_constant_b_reward = 1 - self.normalizing_constant_a_reward * max_reward_value
        #     observation
        #       get maximum ranges from world associated with robot
        vec_distance_max = np.array([self.robot.world.x_max - self.robot.world.x_min, self.robot.world.y_max - self.robot.world.y_min, self.robot.world.z_max - self.robot.world.z_min])
        vec_distance_min = -1 * vec_distance_max
        distance_max = np.linalg.norm(vec_distance_max)
        #       constants
        self.normalizing_constant_a_obs = np.zeros(4)  # 3 for difference vector and 1 for distance itself
        self.normalizing_constant_b_obs = np.zeros(4)  # 3 for difference vector and 1 for distance itself
        self.normalizing_constant_a_obs[:3] = 2 / (vec_distance_max - vec_distance_min)
        self.normalizing_constant_a_obs[3] = 1 / distance_max  # distance only between 0 and 1
        self.normalizing_constant_b_obs[:3] = np.ones(3) - np.multiply(self.normalizing_constant_a_obs[:3], vec_distance_max)
        self.normalizing_constant_b_obs[3] = 1 - self.normalizing_constant_a_obs[3] * distance_max  # this is 0, but keeping it in the code for symmetry

        # placeholders so that we have access in other methods without doing double work
        self.distance = None
        self.position = None
        self.reward_value = 0
        self.shaking = 0
        self.collided = False
        self.timeout = False
        self.out_of_bounds = False
        self.is_success = False
        self.done = False
        self.past_distances = []

        # performance metric name
        self.metric_name = "distance_threshold"

    def get_observation_space_element(self) -> dict:
        if self.add_to_observation_space:
            ret = dict()
            if self.normalize_observations:
                ret[self.output_name ] = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            else:
                high = np.array([self.robot.world.x_max - self.robot.world.x_min, self.robot.world.y_max - self.robot.world.y_min, self.robot.world.z_max - self.robot.world.z_min, 1], dtype=np.float32)
                low = np.array([-self.robot.world.x_max + self.robot.world.x_min, -self.robot.world.y_max + self.robot.world.y_min, -self.robot.world.z_max + self.robot.world.z_min, 0], dtype=np.float32)
                ret[self.output_name ] = Box(low=low, high=high, shape=(4,), dtype=np.float32)

            return ret
        else:
            return {}

    def get_observation(self) -> dict:
        # get the data
        self.position = self.robot.position_rotation_sensor.position
        self.target = self.robot.world.position_targets[self.robot.id]
        dif = self.target - self.position
        self.distance = np.linalg.norm(dif)

        self.past_distances.append(self.distance)
        if len(self.past_distances) > 10:
            self.past_distances.pop(0)

        ret = np.zeros(4)
        ret[:3] = dif
        ret[3] = self.distance
        
        if self.normalize_observations:
            return {self.output_name: np.multiply(self.normalizing_constant_a_obs, ret) + self.normalizing_constant_b_obs} 
        else:
            return {self.output_name: ret}

    def reward(self, step, action):

        reward = 0

        self.out_of_bounds = self._out()
        self.collided = self.robot.world.collision

        shaking = 0
        if len(self.past_distances) >= 10:
            arrow = []
            for i in range(0,9):
                arrow.append(0) if self.past_distances[i + 1] - self.past_distances[i] >= 0 else arrow.append(1)
            for j in range(0,8):
                if arrow[j] != arrow[j+1]:
                    shaking += 1
        self.shaking = shaking
        reward -= shaking * 0.005

        self.is_success = False
        if self.out_of_bounds:
            self.done = True
            reward += self.reward_collision / 2
        elif self.collided:
            self.done = True
            reward += self.reward_collision
        elif self.distance < self.distance_threshold:
            self.done = True
            self.is_success = True
            reward += self.reward_success
        elif step > self.max_steps:
            self.done = True
            self.timeout = True
            reward += self.reward_collision / 10
        else:
            self.done = False
            reward += self.reward_distance_mult * self.distance
        
        self.reward_value = reward
        if self.normalize_rewards:
            self.reward_value = self.normalizing_constant_a_reward * self.reward_value + self.normalizing_constant_b_reward
        
        # return
        return self.reward_value, self.is_success, self.done, self.timeout, self.out_of_bounds    

    def on_env_reset(self, success_rate):
        
        self.timeout = False
        self.is_success = False
        self.is_done = False
        self.collided = False
        self.out_of_bounds = False
        
        # set the distance threshold according to the success of the training
        if self.train: 

            # calculate increment
            ratio_start_end = (self.distance_threshold - self.distance_threshold_end) / (self.distance_threshold_start - self.distance_threshold_end)
            increment = (self.distance_threshold_increment_start - self.distance_threshold_increment_end) * ratio_start_end + self.distance_threshold_increment_end
            if success_rate > self.distance_threshold_change and self.distance_threshold > self.distance_threshold_end:
                self.distance_threshold -= increment
            elif success_rate < self.distance_threshold_change and self.distance_threshold < self.distance_threshold_start:
                #self.distance_threshold += increment / 25  # upwards movement should be slower # DISABLED
                pass
            if self.distance_threshold > self.distance_threshold_start:
                self.distance_threshold = self.distance_threshold_start
            if self.distance_threshold < self.distance_threshold_end:
                self.distance_threshold = self.distance_threshold_end

        return self.metric_name, self.distance_threshold, True, True

    def build_visual_aux(self):
        # build a sphere of distance_threshold size around the target
        self.target = self.robot.world.position_targets[self.robot.id]
        pyb.createMultiBody(baseMass=0,
                            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=self.distance_threshold, rgbaColor=[0, 1, 0, 1]),
                            basePosition=self.target)

    def get_data_for_logging(self) -> dict:
        logging_dict = dict()

        logging_dict["shaking_" + self.robot.name] = self.shaking
        logging_dict["reward_" + self.robot.name] = self.reward_value
        logging_dict["distance_" + self.robot.name] = self.distance
        logging_dict["distance_threshold_" + self.robot.name] = self.distance_threshold

        return logging_dict

    ###################
    # utility methods #
    ###################

    def _out(self):
        
        x, y, z = self.position
        if x > self.robot.world.x_max or x < self.robot.world.x_min:
            return True
        elif y > self.robot.world.y_max or y < self.robot.world.y_min:
            return True
        elif z > self.robot.world.z_max or z < self.robot.world.z_min:
            return True
        return False