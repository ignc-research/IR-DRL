from gym_env.environment import ModularDRLEnv
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from callbacks.callbacks import MoreLoggingCustomCallback
import torch
from os.path import isdir
import numpy as np

import argparse

# here the argparser will read in the config file in the future
# and put the settings into the script_parameters dict
# for now all the settings are done by hand here

script_parameters = {
    "train": True,
    "logging": 1,  # 0: no logging at all, 1: console output on episode end (default as before), 2: same as one 1 + entire log for episode put into txt file at episode end
    "timesteps": 15e6,
    "save_freq": 3e4,
    "save_folder": "./models/weights",
    "save_name": "PPO_test",  # name for the model file, this will get automated later on
    "num_envs": 16,
    "joint_control": False,  # robot controlled by inverse kinematics or by directly acting on the joint angles
    "normalize_observations": False,
    "normalize_rewards": False,
    "gamma": 0.995,
    "tensorboard_folder": "./models/tensorboard_logs/",
    "custom_policy": None,  # custom NN sizes, e.g. dict(activation_fn=torch.nn.ReLU, net_arch=[256, dict(vf=[256, 256], pi=[128, 128])])
    "ppo_steps": 1024,  # steps per env until PPO updates
    "batch_size": 2048,  # batch size for the ppo updates
    "load_model": False,  # set to True when loading an existing model 
    "model_path": './models/weights/PPO_test',  # path for the model when loading one, also used for the eval model when train is set to False
}

# do not change the env_configs below
env_config_train = {
    "train": True,
    "logging": 1,
    "joint_control": script_parameters["joint_control"],
    "normalize_observations": script_parameters["normalize_observations"],
    "normalize_rewards": script_parameters["normalize_rewards"],
    "display": False,
    "display_extra": False
}

env_config_eval = {
    "train": False,
    "logging": script_parameters["logging"],
    "joint_control": script_parameters["joint_control"],
    "normalize_observations": script_parameters["normalize_observations"],
    "normalize_rewards": script_parameters["normalize_rewards"],
    "display": True,
    "display_extra": True
}


if __name__ == "__main__":
    if script_parameters["train"]:
        
        def return_train_env_outer():
            def return_train_env_inner():
                env = ModularDRLEnv(env_config_train)
                return env
            return return_train_env_inner
        
        # create parallel envs
        envs = SubprocVecEnv([return_train_env_outer() for i in range(script_parameters["num_envs"])])

        # callbacks
        checkpoint_callback = CheckpointCallback(save_freq=script_parameters["save_freq"], save_path=script_parameters["save_folder"], name_prefix=script_parameters["save_name"])
        more_logging_callback = MoreLoggingCustomCallback()

        callback = CallbackList([checkpoint_callback, more_logging_callback])

        # create or load model
        if not script_parameters["load_model"]:
            model = PPO("MultiInputPolicy", envs, policy_kwargs=script_parameters["custom_policy"], verbose=1, gamma=script_parameters["gamma"], tensorboard_log=script_parameters["tensorboard_folder"], n_steps=script_parameters["ppo_steps"], batch_size=script_parameters["batch_size"])
        else:
            model = PPO.load(script_parameters["model_path"], env=envs, tensorboard_log=script_parameters["tensorboard_folder"])
            # needs to be set on my pc when loading a model, dont know why, might not be needed on yours
            model.policy.optimizer.param_groups[0]["capturable"] = True

        model.learn(total_timesteps=script_parameters["timesteps"], callback=callback, tb_log_name=script_parameters["save_name"], reset_num_timesteps=False)

    else:
        env = ModularDRLEnv(env_config_eval)
        #model = PPO.load(script_parameters["model_path"], env=env)
        model = PPO("MultiInputPolicy", env, policy_kwargs=script_parameters["custom_policy"], verbose=1, gamma=script_parameters["gamma"], tensorboard_log=script_parameters["tensorboard_folder"], n_steps=script_parameters["ppo_steps"])

        for i in range(30):
            obs = env.reset()
            done = False
            while not done:
                act = model.predict(obs)[0]
                obs, reward, done, info = env.step(act)
