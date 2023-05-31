# this is a file to host random assorted scripts that were useful at some place in the code

import numpy as np

def regular_equidistant_sphere_points(N, r):
    """
    Creates ~N regularly and equidistantly distributed points on the surface of a sphere with radius r.
    see: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    return_points = []
    n_count = 0
    a = (4 * np.pi * r**2) / N
    d = np.sqrt(a)
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta
    d_phi = a / d_theta
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            return_points.append([np.sin(theta) * np.cos(phi) * r, np.sin(theta) * np.sin(phi) * r, np.cos(theta) * r])
            n_count += 1
    
    return np.array(return_points)

def fibonacci_sphere(N):
    """
    Creates N regularly and equidistantly distributed points on the surface of a sphere with radius 1.
    see: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def analyse_obs_spaces(env_obs_space, model_obs_space):
    # method to check which elements of the observation space don't match when loading a model fails
    entries_only_in_env = []
    entries_only_in_model = []
    entries_in_both_but_with_different_values = []
    for key in env_obs_space.keys():
        if key not in model_obs_space.keys():
            entries_only_in_env.append(key)
        elif key in model_obs_space.keys() and model_obs_space[key]!=env_obs_space[key]:
            entries_in_both_but_with_different_values.append((key, model_obs_space[key], env_obs_space[key]))
    for key in model_obs_space.keys():
        if key not in env_obs_space.keys():
            entries_only_in_model.append(key)
    print("[Error] Your model failed to load due to incompatible observation spaces!")
    print("This is a list of all elements that were only found in the env observation space:")
    print("#"*15)
    for entry in entries_only_in_env:
        print(entry)
    print("#"*15)
    print("This is a list of all elements that were only found in the model's observation space:")
    print("#"*15)
    for entry in entries_only_in_model:
        print(entry)
    print("#"*15)
    print("This is a list of all elements that were found in both, but have mismatching definitions:")
    print("#"*15)
    for entry in entries_in_both_but_with_different_values:
        print("Entry: ", entry[0])
        print("Value in model: ", entry[1])
        print("Value in env: ", entry[2])
    print("#"*15)