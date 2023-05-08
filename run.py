# parse command line args
from argparse import ArgumentParser
from modular_drl_env.util.configparser import parse_config

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
from modular_drl_env.gym_env.environment import ModularDRLEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from modular_drl_env.callbacks.callbacks import MoreLoggingCustomCallback
from time import sleep
import signal
import sys

# import the RL algorithms
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from sb3_contrib import RecurrentPPO, AttentionPPO

algo_map = {
    "PPO": (PPO, "MultiInputPolicy"),
    "AttentionPPO": (AttentionPPO, "MultiInputAttnPolicy"),
    "RecurrentPPO": (RecurrentPPO, "MultiInputLstmPolicy"),
    "TD3": (TD3, "MultiInputPolicy"),
    "SAC": (SAC, "MultiInputPolicy"),
    "A2C": (A2C, "MultiInputPolicy"),
    "DDPG": (DDPG, "MultiInputPolicy")
}

if __name__ == "__main__":
    algorithm = algo_map[run_config["algorithm"]["type"]][0]
    policy = algo_map[run_config["algorithm"]["type"]][1]
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
        checkpoint_callback = CheckpointCallback(save_freq=run_config["save_freq"], save_path=run_config["save_folder"] + "/" + run_config["algorithm"]["type"] + "_" + run_config["save_name"], name_prefix="model")
        more_logging_callback = MoreLoggingCustomCallback()

        callback = CallbackList([checkpoint_callback, more_logging_callback])

        # create or load model
        if not run_config["algorithm"]["load_model"]:       
            model = algorithm(policy, envs, policy_kwargs=run_config["custom_policy"], verbose=1, gamma=run_config["algorithm"]["gamma"], learning_rate=run_config["algorithm"]["learning_rate"], tensorboard_log="./models/tensorboard_logs", **run_config["algorithm"]["config"])
            print(model.policy)
        else:
            model = algorithm.load(run_config["algorithm"]["model_path"], env=envs, tensorboard_log="./models/tensorboard_logs")
            # needs to be set on some PCs when loading a model, dont know why, might not be needed on yours
            if run_config["algorithm"]["type"] == "PPO":
                model.policy.optimizer.param_groups[0]["capturable"] = True
        
        # signal handler, allows us to save a model if we interrupt training by ctrl + c
        # code copied from https://git.tu-berlin.de/erik.fischer98/autonomous-agents/-/blob/main/franka_move/franka_train.py
        def signal_handler(sig, frame):
                model.save(run_config["save_folder"] + "/" + run_config["algorithm"]["type"] + "_" + run_config["save_name"] + "/model_interrupt")
                sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        
        # perform learning
        model.learn(total_timesteps=run_config["timesteps"], callback=callback, tb_log_name=run_config["algorithm"]["type"] + "_" + run_config["save_name"], reset_num_timesteps=False)

    else:
        env_config["env_id"] = 0
        env = ModularDRLEnv(env_config)
        if not run_config["algorithm"]["load_model"]:
            # no extra case for recurrent model here, this would act exatcly the same way here as a new PPO does
            # we use PPO here, but as its completely untrained might just be any model really
            model = PPO("MultiInputPolicy", env, verbose=1)
        else:
            model = algorithm.load(run_config["algorithm"]["model_path"], env=env)

        while True:
            obs = env.reset()
            done = False
            # episode start signals for recurrent model
            episode_start = True
            state = None
            steps = 0
            next_steps = 1
            offset = 0
            while not done:
                sleep(run_config["display_delay"])
                act, state = model.predict(obs, state=state, episode_start=episode_start)
                obs, reward, done, info = env.step(act)
                episode_starts = done
                steps += 1
                if args.debug:
                    print("--------------")
                    print("Step:")
                    print(steps)
                    print("Env observation:")
                    print(obs)
                    print("Agent action:")
                    print(act)
                    if (steps - offset) % next_steps == 0:
                        inp = input("Press any button to continue one step, enter a number to continue that number of steps or press r to reset the episode:\n")
                        if inp == "r":
                            done = True
                        else:
                            try:
                                next_steps = int(inp)
                                next_steps = 1 if next_steps == 0 else next_steps
                                offset = steps
                            except ValueError:
                                next_steps = 1


