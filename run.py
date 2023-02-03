# parse command line args
from argparse import ArgumentParser
from util.configparser import parse_config

# parse the three arguments
parser = ArgumentParser(prog = "Modular DRL Robot Gym Env",
                        description = "Builds and runs a modular gym env for simulated robots using a config file.")
parser.add_argument("configfile", help="Path to the config yaml you want to use.")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--train", action="store_true", help="Runs the env in train mode.")
mode.add_argument("--eval", action="store_true", help="Runs the env in eval mode.")
mode.add_argument("--debug", action="store_true", help="Runs the env in eval mode but stops on every step for user input and prints the observations.")

args = parser.parse_args()

# fetch the config and parse it into python objects
run_config, env_config = parse_config(args.configfile, args.train)

# we import the rest here because this takes quite some time and we want the arg parsing to be fast and responsive
from gym_env.environment import ModularDRLEnv
from stable_baselines3 import PPO, TD3, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from callbacks.callbacks import MoreLoggingCustomCallback
import torch
from os.path import isdir
import numpy as np
import zennit as z
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from typing import Callable          
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import preprocess_obs, is_image_space
from zennit.rules import Epsilon, Gamma
from time import sleep


#pos goal : 4, joints : 6 , pos and rot : 4, lidar : 25, camera : 256

        
        

def find_in_features(seq):
    for sub in seq:
        if type(sub) is torch.nn.Sequential:
            return find_in_features(sub)
        else:
            return sub.in_features

#from explanability import ExplainPPO, VisualizeExplanations

if __name__ == "__main__":
    if run_config["train"]:
        
        def return_train_env_outer(i):
            def return_train_env_inner():
                env_config["env_id"] = i
                env = ModularDRLEnv(env_config)
                return env
            return return_train_env_inner
        
        # create parallel envs
        envs = SubprocVecEnv([return_train_env_outer(i) for i in range(run_config["num_envs"])])

        # callbacks
        checkpoint_callback = CheckpointCallback(save_freq=run_config["save_freq"], save_path=run_config["save_folder"] + "/" + run_config["save_name"], name_prefix="model")
        more_logging_callback = MoreLoggingCustomCallback()

        callback = CallbackList([checkpoint_callback, more_logging_callback])

        # create or load model
        if not run_config["load_model"]:
            if run_config["recurrent"]:
                model = RecurrentPPO("MultiInputLstmPolicy", envs, policy_kwargs=run_config["custom_policy"], verbose=1, gamma=run_config["gamma"], tensorboard_log=run_config["tensorboard_folder"], n_steps=run_config["ppo_steps"], batch_size=run_config["batch_size"])
            else:
                model = PPO("MultiInputPolicy", envs, policy_kwargs=run_config["custom_policy"], verbose=1, gamma=run_config["gamma"], tensorboard_log=run_config["tensorboard_folder"], n_steps=run_config["ppo_steps"], batch_size=run_config["batch_size"])
            print(model.policy)
        else:
            if run_config["recurrent"]:
                model = RecurrentPPO.load(run_config["model_path"], env=envs, tensorboard_log=run_config["tensorboard_folder"])
            else:
                model = PPO.load(run_config["model_path"], env=envs, tensorboard_log=run_config["tensorboard_folder"])
            # needs to be set on my pc when loading a model, dont know why, might not be needed on yours
            model.policy.optimizer.param_groups[0]["capturable"] = True

        model.learn(total_timesteps=run_config["timesteps"], callback=callback, tb_log_name=run_config["save_name"], reset_num_timesteps=False)

    else:
        env_config["env_id"] = 0
        env = ModularDRLEnv(env_config)
        if not run_config["load_model"]:
            # no extra case for recurrent model here, this would act exatcly the same way here as a new PPO does
            model = PPO("MultiInputPolicy", env, verbose=1)
        else:
            if run_config["recurrent"]:
                model = RecurrentPPO.load(run_config["model_path"], env=env)
            else:
                model = PPO.load(run_config["model_path"], env=env)

        #explainer = ExplainPPO(env, model, extractor_bias= 'camera')
        #exp_visualizer = VisualizeExplanations(explainer, type_of_data= 'rgbd')
        while True:
            obs = env.reset()
            #exp_visualizer.close_open_figs()
            #fig, axs = exp_visualizer.start_imshow_from_obs(obs, value_or_action='action')
            done = False
            # episode start signals for recurrent model
            episode_start = True if run_config["recurrent"] else None
            state = None
            while not done:
                sleep(run_config["display_delay"])
                act, state = model.predict(obs, state=(state if run_config["recurrent"] else None), episode_start=episode_start)
                obs, reward, done, info = env.step(act)
                episode_starts = done
                if args.debug:
                    print("--------------")
                    print("Env observation:")
                    print(obs)
                    print("Agent action:")
                    print(act)
                    inp = input("Press any button to continue with next step or press r to reset the episode:\n")
                    if inp == "r":
                        done = True
                #exp_visualizer.update_imshow_from_obs(obs, fig, axs)


