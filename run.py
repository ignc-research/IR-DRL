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
mode.add_argument("--planner", default="", help="Runs the env without a DRL agent but with a planner instead. Needs a compatible world. For now this will also only work with one robot.")

args = parser.parse_args()

# fetch the config and parse it into python objects
run_config, env_config = parse_config(args.configfile, args.train)

# we import the rest here because this takes quite some time and we want the arg parsing to be fast and responsive
from modular_drl_env.gym_env.environment import ModularDRLEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from modular_drl_env.callbacks.callbacks import MoreLoggingCustomCallback
from modular_drl_env.util.misc import analyse_obs_spaces
from time import sleep
import signal
import sys

# import the RL algorithms
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from sb3_contrib import RecurrentPPO, AttentionPPO

# import planners
from modular_drl_env.planner.planner_implementations import *

algo_map = {
    "PPO": (PPO, "MultiInputPolicy"),
    "AttentionPPO": (AttentionPPO, "MultiInputAttnPolicy"),
    "RecurrentPPO": (RecurrentPPO, "MultiInputLstmPolicy"),
    "TD3": (TD3, "MultiInputPolicy"),
    "SAC": (SAC, "MultiInputPolicy"),
    "A2C": (A2C, "MultiInputPolicy"),
    "DDPG": (DDPG, "MultiInputPolicy")
}
planner_map = {
    "RRT" : RRT,
    "BiRRT": BiRRT,
    "RRT*": RRTStar,
    "PRM": PRM
}

noise_map = {
    "OrnsteinUhlenbeck": OrnsteinUhlenbeckActionNoise,
    "Gaussian": NormalActionNoise
}

if __name__ == "__main__":
    if args.planner == "":
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
                if "action_noise" in run_config["algorithm"]["config"]:
                    import numpy as np
                    run_config["algorithm"]["config"]["action_noise"] = noise_map[run_config["algorithm"]["config"]["action_noise"][0]](mean=np.zeros(run_config["algorithm"]["config"]["action_noise"][1]), sigma=np.ones(run_config["algorithm"]["config"]["action_noise"][1]))
                model = algorithm(policy, envs, policy_kwargs=run_config["custom_policy"], verbose=1, tensorboard_log="./models/tensorboard_logs", **run_config["algorithm"]["config"])
                print(model.policy)
            else:
                model = algorithm.load(run_config["algorithm"]["model_path"], env=envs, tensorboard_log="./models/tensorboard_logs", custom_objects=run_config["algorithm"]["config"])
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
                # this try except will fail if the env's observation space doesn't match with the one from the model
                # the standard error message by sb3 isn't very helpful, so this piece of code will pinpoint exactly what
                try:
                    model = algorithm.load(run_config["algorithm"]["model_path"], env=env)
                except ValueError:
                    from stable_baselines3.common.save_util import load_from_zip_file
                    data, _, _ = load_from_zip_file(
                        run_config["algorithm"]["model_path"]
                    )
                    analyse_obs_spaces(env.observation_space, data["observation_space"])
                    exit(0)
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
                    act, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
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
                        print("Reward:")
                        print(reward)
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
    else:
        env_config["env_id"] = 0
        env = ModularDRLEnv(env_config)
        robot = env.robots[0]
        # overwrite control mode to trajectory
        robot.control_mode = 3  
        # overwrite goal to standard position goal, we need no other one
        from modular_drl_env.goal.goal_implementations.position_collision import PositionCollisionGoal
        goal = PositionCollisionGoal(robot, False, False, False, False, 10000, False, done_on_oob=False)
        env.goals[0] = goal
        robot.goal = goal
        goal.distance_threshold = 0.055
        # remove all sensors beside the mandatory ones
        env.sensors = env.sensors[:2] 
        planner = planner_map[args.planner](robot)
        import numpy as np
        act = np.zeros(len(robot.controlled_joints_ids))

        from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u
        while True:
            obs = env.reset()
            trajectory = np.array(planner.plan(env.world.joints_targets[0], env.world.active_objects))
            trajectory_xyz = []
            for waypoint in trajectory:
                robot.moveto_joints(waypoint, False)
                xyz, _, _, _ = pyb_u.get_link_state(robot.object_id, robot.end_effector_link_id)
                trajectory_xyz.append(xyz)
            robot.moveto_joints(trajectory[0], False)
            trajectory_xyz = np.array(trajectory_xyz)
            done = False
            idx = 0
            while not done:
                sleep(run_config["display_delay"])
                robot.control_target = trajectory[idx]
                _, _, done, info = env.step(act)
                ee_position = robot.position_rotation_sensor.position
                distances_to_ee = np.linalg.norm(trajectory_xyz -  ee_position, axis=1)
                sphere_mask = distances_to_ee < 0.015
                if sphere_mask.any() == False:  
                    min_distance = np.min(distances_to_ee)
                    index_of_that_waypoint = np.where(distances_to_ee == min_distance)[0][0]
                    idx = max(idx, index_of_that_waypoint)  # keep old waypoint if it's further up in the trajectory
                else:
                    idxs = np.array(list(range(len(trajectory))))[sphere_mask]
                    idx = max(idx, max(idxs))

                if done:
                    break
