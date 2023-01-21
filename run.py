from gym_env.environment import ModularDRLEnv
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from callbacks.callbacks import MoreLoggingCustomCallback
import torch
from explanability import ExplainPPO, VisualizeExplanations


# here the argparser will read in the config file in the future
# and put the settings into the script_parameters dict
# for now all the settings are done by hand here

script_parameters = {
    "train": True,
    "logging": 1,  # 0: no logging at all, 1: console output on episode end (default as before), 2: same as one 1 + entire log for episode put into csv file at episode end; if max_episodes is not -1 then the csv will contain the data for all episodes
    "timesteps": 15e6,
    "max_steps_per_episode": 1024,
    "max_episodes": 30,  # num episodes for eval
    "save_freq": 3e4,
    "save_folder": "./models/weights",
    "save_name": "PPO_floating_fe_0",  # name for the model file, this will get automated later on
    "num_envs": 48,
    "use_physics_sim": True,  # use actual physics sim or ignore forces and teleport robot to desired poses
    "control_mode": 1,  # robot controlled by inverse kinematics (0), joint angles (1) or joint velocities (2)
    "sim_step": 1 / 240,  # seconds that pass per env step
    "normalize_observations": False,
    "normalize_rewards": False,
    "gamma": 0.9918,
    "dist_threshold_overwrite": None,  # use this when continuing training to set the distance threhsold to the value that your agent had already reached
    "stat_buffer_size": 25,  # number of past episodes for averaging success metrics
    "tensorboard_folder": "./models/tensorboard_logs/",
    "custom_policy": None,  # custom NN sizes, e.g. dict(activation_fn=torch.nn.ReLU, net_arch=[256, dict(vf=[256, 256], pi=[128, 128])])
    "ppo_steps": 1024,  # steps per env until PPO updates
    "batch_size": 512,  # batch size for the ppo updates
    "load_model": False,  # set to True when loading an existing model 
    "model_path": './models_bennoEnv/weights/PPO_bodycam_0_8640000_steps',  # path for the model when loading one, also used for the eval model when train is set to False
}

# do not change the env_configs below
env_config_train = {
    "train": True,
    "logging": 1,
    "max_steps_per_episode": script_parameters["max_steps_per_episode"],
    "max_episodes": -1,
    "sim_step": script_parameters["sim_step"],
    "stat_buffer_size": script_parameters["stat_buffer_size"],
    "use_physics_sim": script_parameters["use_physics_sim"],
    "control_mode": script_parameters["control_mode"],
    "normalize_observations": script_parameters["normalize_observations"],
    "normalize_rewards": script_parameters["normalize_rewards"],
    "dist_threshold_overwrite": script_parameters["dist_threshold_overwrite"],
    "display": False,
    "display_extra": False
}

env_config_eval = {
    "train": False,
    "logging": script_parameters["logging"],
    "max_steps_per_episode": script_parameters["max_steps_per_episode"],
    "max_episodes": script_parameters["max_episodes"],
    "sim_step": script_parameters["sim_step"],
    "stat_buffer_size": script_parameters["stat_buffer_size"],
    "use_physics_sim": script_parameters["use_physics_sim"],
    "control_mode": script_parameters["control_mode"],
    "normalize_observations": script_parameters["normalize_observations"],
    "normalize_rewards": script_parameters["normalize_rewards"],
    "dist_threshold_overwrite": script_parameters["dist_threshold_overwrite"],
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
        if not script_parameters["load_model"]:
            model = PPO("MultiInputPolicy", env, policy_kwargs=script_parameters["custom_policy"], verbose=1, gamma=script_parameters["gamma"], tensorboard_log=script_parameters["tensorboard_folder"], n_steps=script_parameters["ppo_steps"])
            print('new model')
        else:
            model = PPO.load(script_parameters["model_path"], env=env)

        explainer = ExplainPPO(env, model, extractor_bias= 'camera')
        exp_visualizer = VisualizeExplanations(explainer, type_of_data= 'rgbd')


        while True:
            obs = env.reset()
            exp_visualizer.close_open_figs()
            fig, axs = exp_visualizer.start_imshow_from_obs(obs, value_or_action='action', grad_outputs=torch.eye(6)[[0]])
            done = False
            while not done:
                act = model.predict(obs)[0]
                obs, reward, done, info = env.step(act)
                exp_visualizer.update_imshow_from_obs(obs, fig, axs, grad_outputs=torch.eye(6)[[0]])


