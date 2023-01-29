import yaml
import numpy as np
import torch as th
import pybullet as pyb

def walk_dict_and_convert_to_our_format(node):
    # walk through the nested dictionaries and lists
    for key, item in node.items():
        # deal with lists
        if type(item) == list:
            # skip empty lists
            if len(item) == 0:
                continue
            # if the list is a part of a special category, it is composed of dicts
            # then we must traverse deeper
            # robots: robots field for robot definition
            # sensors: sensors field for sensor defitinion, both for robot bound and independet sensors
            # obstacles: obstacles for the generator world type
            if key == "robots" or key == "sensors" or key == "obstacles":
                for element in item:
                    walk_dict_and_convert_to_our_format(element)
            # deal with rotations and angles
            if "rotation" in key or "orientation" in key or "angle" in key:
                # case 1: it's a single orientation (e.g. orientation of an obstacle)
                if type(item[0]) == float or type(item[0]) == int:
                    # convert to radians and quaternion
                    item = [x * np.pi/180 for x in item]
                    if not "angle" in key:
                        item = pyb.getQuaternionFromEuler(item)
                    node[key] = item
                # case 2: it's a list of orientations (e.g. a trajectory of orientations)
                elif type(item[0]) == list and type(item[0][0]) == float or type(item[0][0]) == int:
                    item = [[x * np.pi/180 for x in inner] for inner in item]
                    if not "angle" in key:
                        item = [pyb.getQuaternionFromEuler(inner) for inner in item]
                    node[key] = item
                # case 3: it's a list of lists of orientations (e.g. multiple trajectories of orientations)
                # note: this will fail/potentially crash if the first of these trajectories is empty and others are not
                elif (type(item[0][0]) == list and type(item[0][0][0]) == float or type(item[0][0][0]) == int):
                    item = [[[x * np.pi/180 for x in inner] for inner in outer] for outer in item]
                    if not "angle" in key:
                        item = item = [[pyb.getQuaternionFromEuler(inner) for inner in outer] for outer in item]
                    node[key] = item
                # pls no more nesting than that
        elif type(item) == dict:
            # go a level depper
            walk_dict_and_convert_to_our_format(item)
        elif item == "None":
            # replace written Nones
            node[key] = None

    return
    for key, item in node.items():
        if key == "robots" or key == "sensors":
            for element in item:
                walk_dict_and_convert_to_our_format(element)
        if type(item) == dict:
            walk_dict_and_convert_to_our_format(item)
        elif type(item) == list:
            #node[key] = np.array(item)
            if len(item) == 0:
                continue
            if "orientation" in key or "rotation" in key or "angle" in key:
                # convert to radians
                # check if we have a nested list one or two levels deep
                if len(item) > 0 and type(item[0]) == list:
                    # check for next level (maximum level of nested lists we allow)
                    if len(item[0]) > 0 and item[0][0] == list:  # two-nested
                        item = [[[x * np.pi/180 for x in inner] for inner in outer] for outer in item]
                        if "orientation" in key or "rotation" in key:
                            item = [[pyb.getQuaternionFromEuler(inner) for inner in outer] for outer in item]
                        node[key] = item
                    else:  # one-nested
                        item = [[x * np.pi/180 for x in inner] for inner in item]
                        if "orientation" in key or "rotation" in key:
                            item = [pyb.getQuaternionFromEuler(inner) for inner in item]
                        node[key] = item
                else:  # not nested
                    item = [x * np.pi/180 for x in item]
                    if "orientation" in key or "rotation" in key:
                        item = pyb.getQuaternionFromEuler(item)
                    node[key] = item
        elif item == "None":
            node[key] = None


def parse_config(filepath, train):
    with open(filepath, "r") as infile:
        config_raw = yaml.safe_load(infile)
    
    # copy keys a layer up
    for key in config_raw["run"]["train"]:
        config_raw["run"][key] = config_raw["run"]["train"][key]

    # convert the dict description of custom policy to actual format used by sb3
    if not config_raw["run"]["train"]["custom_policy"]["use"]:
        config_raw["run"]["custom_policy"] = None
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
        config_raw["run"]["custom_policy"] = pol_dict

    # set some defaults for train or eval
    if train:
        config_raw["run"]["display"] = False
        config_raw["run"]["display_extra"] = False
    else:
        config_raw["run"]["display"] = True
        config_raw["run"]["display_extra"] = True
        config_raw["run"]["display_delay"] = config_raw["run"]["eval"]["display_delay"]

    # set train status
    config_raw["run"]["train"] = train

    # convert all lists to numpy, angles to radians and rpy to quaternion
    walk_dict_and_convert_to_our_format(config_raw["env"])

    env_config = config_raw["env"].copy()
    env_config["train"] = train
    env_config["display"] = True
    env_config["display_extra"] = True
    env_config["max_episodes"] = config_raw["run"]["eval"]["max_episodes"]
    env_config["logging"] = config_raw["run"]["eval"]["logging"]
    if train:
        env_config["max_episodes"] = -1
        env_config["logging"] = 1
        env_config["display"] = False
        env_config["display_extra"] = False

    del config_raw["run"]["eval"]

    run_config = config_raw["run"].copy()

    return run_config, env_config