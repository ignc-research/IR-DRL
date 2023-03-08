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

    srcp = 2 * (w * x + y * z)
    crcp = 1 - 2 * (x * x + y * y)
    r = np.arctan2(srcp, crcp)

    sp = np.sqrt(1 + 2 * (w * y - x * z))
    cp = np.sqrt(1 - 2 * (w * y - x * z))
    p = 2 * np.arctan2(sp, cp) - np.pi / 2

    sycp = 2 * (w * z + x * y)
    cycp = 1 - 2 * (y * y + z * z)
    y = np.arctan2(sycp, cycp)

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
