# parse command line args
import time
import json
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

i = 1
while True:
    obs = testo.reset()

    # for i in range(8):
    #     print(i)
    #     link_pos = np.asarray(pyb.getLinkState(0, i)[4])
    #     add = np.array([0, 0, 1])
    #     pyb.addUserDebugLine(link_pos, link_pos + add)
    #     time.sleep(2)
    #
    # q_init = obs["joints_angles_ur5_1"] * np.pi
    # target = testo.world.position_targets[0]
    # rrt_planner = RRT(robot_id=0,
    #                   robot_joints=[1, 2, 3, 4, 5, 6],
    #                   end_effector_index=7,
    #                   obstacles=[2])
    # traj = rrt_planner.compute_trajectory(q_init, target)
    # if traj is not None:
    #     with open("trajectories.json", "r") as f:
    #         data = json.load(f)
    #
    #     with open("trajectories.json", "w") as f:
    #         data["trajectories"] = data["trajectories"] + [traj.tolist()]
    #         json.dump(data, f)

    done = False
    while not done:
        obs, reward, done, info = testo.step(testo.action_space.sample())






