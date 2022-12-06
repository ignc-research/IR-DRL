import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from env import Env

params = {
    'is_render': True, 
    'is_good_view': True,
    'is_train' : False,
    'show_boundary' : True,
    'add_moving_obstacle' : False,
    'moving_obstacle_speed' : 0.15,
    'moving_init_direction' : -1,
    'moving_init_axis' : 0,
    'workspace' : [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
    'max_steps_one_episode' : 1024,
    'num_obstacles' : 3,
    'prob_obstacles' : 0.8,
    'obstacle_box_size' : [0.04,0.04,0.002],
    'obstacle_sphere_radius' : 0.04       
}


if __name__=='__main__':

    env = Env(
        is_render=params['is_render'],
        is_good_view=params['is_good_view'],
        is_train=params['is_train'],
        show_boundary=params['show_boundary'],
        add_moving_obstacle=params['add_moving_obstacle'],
        moving_obstacle_speed=params['moving_obstacle_speed'],
        moving_init_direction=params['moving_init_direction'],
        moving_init_axis=params['moving_init_axis'],
        workspace=params['workspace'],
        max_steps_one_episode=params['max_steps_one_episode'],
        num_obstacles=params['num_obstacles'],
        prob_obstacles=params['prob_obstacles'],
        obstacle_box_size=params['obstacle_box_size'],
        obstacle_sphere_radius=params['obstacle_sphere_radius']
        )
    # load drl model
    model = PPO.load('./ur5/StaticEnv/models/reach_ppo_ckp_logs/reach_2457600_steps', env=env)

    while True:
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
