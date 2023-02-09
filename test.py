# parse command line args
import time
from argparse import ArgumentParser
from configs.configparser import parse_config
from gym_env.environment import ModularDRLEnv
import pybullet as pyb
import numpy as np
from motion_planner.rrt_planner import RRT
# parse the three arguments
parser = ArgumentParser(prog="Modular DRL Robot Gym Env",
                        description = "Builds and runs a modular gym env for simulated robots using a config file.")
parser.add_argument("configfile", help="Path to the config yaml you want to use.")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--train", action="store_true", help="Runs the env in train mode.")
mode.add_argument("--eval", action="store_true", help="Runs the env in eval mode.")
args = parser.parse_args()

# fetch the config and parse it into python objects
run_config, env_config = parse_config(args.configfile, args.train)
testo = ModularDRLEnv(env_config)


while True:
    obs = testo.reset()
    done = False
    while not done:
        obs, reward, done, info = testo.step(testo.action_space.sample())
        target = testo.world.position_targets[0]
        robot = testo.robots[0]

        # see if there is a collision free trajectory towards the goal
        #rrt = RRT(robot.id, robot.joints_ids, [2], 7, max_iterations=10000, f=3)
        #joint_pos = rrt.compute_trajectory(robot.resting_pose_angles, target)
        #if joint_pos is not None:
        #    print(target)
        #    with open("targets.txt", "a") as f:
         #       f.write(f"{target[0]} {target[1]} {target[2]}\n")





