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

regular_equidistant_sphere_points(5, 1)