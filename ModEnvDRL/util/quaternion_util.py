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