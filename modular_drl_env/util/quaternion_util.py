import math

import numpy as np

# small collection of helper functions for quaternions with format x, y, z, w
# (most pip libraries work with w x y z)

def quaternion_invert(quat):
    
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])

def quaternion_multiply(quat1, quat2):
    
    x0, y0, z0, w0 = quat1
    x1, y1, z1, w1 = quat2

    x2 = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y2 = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z2 = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    w2 = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

    return np.array([x2, y2, z2, w2])

def rotate_vector(vector, quat):
    
    help_vector = np.array([vector[0], vector[1], vector[2], 0])
    quat_inv = quaternion_invert(quat)

    rotated_vector = quaternion_multiply(quaternion_multiply(quat, help_vector), quat_inv)

    return rotated_vector[:3]

def quaternion_similarity(quat1, quat2):
    """
    Measure of similarity between two quaternions via the angle distance between the two
    """
    return 1 - np.arccos(np.clip(2 * np.dot(quat1, quat2)**2 - 1, -1, 1))/np.pi

def quaternion_apx_eq(quat1, quat2, thresh=5e-2):
    return  (1 - quaternion_similarity(quat1, quat2)) < thresh

def quaternion_to_rpy(quat):
    x, y, z, w = quat

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    r = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    p = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    y = math.atan2(t3, t4)

    return np.array([r, p, y])

def rpy_to_quaternion(rpy):
    r, p, y = rpy
    
    cr = np.cos(r * 0.5)
    sr = np.sin(r * 0.5)
    cp = np.cos(p * 0.5)
    sp = np.sin(p * 0.5)
    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w])

def quaternion_to_matrix(quat):
    x, y, z, w = quat

    matrix = np.zeros((3,3))
    matrix[0, 0] = 1 - 2 * y**2 - 2 * z**2
    matrix[0, 1] = 2 * x * y - 2 * z * w
    matrix[0, 2] = 2 * x * z + 2 * y * w
    matrix[1, 0] = 2 * x * y + 2 * z * w
    matrix[1, 1] = 1 - 2 * x**2 - 2 * z**2
    matrix[1, 2] = 2 * y * z - 2 * x * w
    matrix[2, 0] = 2 * x * z - 2 * y * w
    matrix[2, 1] = 2 * y * z + 2 * x * w
    matrix[2, 2] = 1 - 2 * x**2 - 2 * y**2

    return matrix

def matrix_to_quaternion(mat):

    x = mat[2][1] - mat[1][2]
    y = mat[0][2] - mat[2][0]
    z = mat[1][0] - mat[0][1]
    w = np.sqrt(1.0 + mat[0][0] + mat[1][1] + mat[2][2])

    return np.array([x, y, z, w])
    
