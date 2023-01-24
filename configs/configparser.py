import yaml
import numpy as np
import torch as th
import pybullet as pyb

def walk_dict_and_convert_to_our_format(node):
    for key, item in node.items():
        if type(item) == dict:
            walk_dict_and_convert_to_our_format(item)
        elif type(item) == list:
            #node[key] = np.array(item)
            if "orientation" in key or "rotation" in key or "angle" in key:
                # convert to radians
                item = [x * np.pi/180 for x in item]
                if "orientation" in key or "rotation" in key:
                    item = pyb.getQuaternionFromEuler(item)
            node[key] = np.array(item)


def parse_config(filepath, train):
    with open(filepath, "r") as infile:
        config_raw = yaml.safe_load(infile)

    # set train status
    config_raw["run"]["train_status"] = train


    # convert the dict description of custom policy to actual format used by sb3
    if not config_raw["run"]["train"]["custom_policy"]["use"]:
        config_raw["run"]["train"]["custom_policy"] = None
    else:
        # get activation function
        if config_raw["run"]["train"]["custom_policy"]["activation_function"] == "ReLU":
            activation_function = th.nn.ReLU
        elif config_raw["run"]["train"]["custom_policy"]["activation_function"] == "tanh":
            activation_function = th.nn.Tanh
        else:
            raise Exception("Unsupported activation function!")
        pol_dict = dict(activation_fn=activation_function)
        net_arch = []
        vf_pi_dict = dict(vf=[], pi=[])
        for key in config_raw["run"]["train"]["custom_policy"]["layers"]:
            if "layer" in key:
                net_arch.append(config_raw["run"]["train"]["custom_policy"]["layers"][key])
            if "value" in key:
                for key2 in config_raw["run"]["train"]["custom_policy"]["layers"][key]:
                    vf_pi_dict["vf"].append(config_raw["run"]["train"]["custom_policy"]["layers"][key][key2])
            elif "policy" in key:
                for key2 in config_raw["run"]["train"]["custom_policy"]["layers"][key]:
                    vf_pi_dict["pi"].append(config_raw["run"]["train"]["custom_policy"]["layers"][key][key2])
        net_arch.append(vf_pi_dict)
        pol_dict["net_arch"] = net_arch
        config_raw["run"]["train"]["custom_policy"] = pol_dict

          

    run_config = config_raw["run"].copy()

    # convert all lists to numpy
    walk_dict_and_convert_to_our_format(config_raw["env"])

    env_config = config_raw["env"].copy()

    return run_config, env_config