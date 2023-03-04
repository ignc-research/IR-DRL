import torch
import numpy as np
from typing import Union, List




class CameraRailBoundingBox:
    """
    Circular path aligned on z-axis with bounding box

    param: x_low: lower bound of box, x-axis
    param: x_high: higher bound of box x-axis
    param: y_low: lower bound of box y-axis
    param: y_high: higher bound of box y-axis
    param: z_height: offset in z-axis
    param: phi: starting position in the circle
    """
    def __init__(self, x_low, x_high, y_low, y_high, z_height, phi= 0): # phi in RAD
        x_mid = (x_high + x_low)/2
        y_mid = (y_high + y_low)/2
        self.center = np.array([x_mid, y_mid])
        self.radius = max((x_high - x_low)/2, (y_high - y_low)/2)
        self.z = z_height
        self.phi = phi
        self.position = self._get_coords()
        self.vel = 0


    def get_coords(self, d_phi, factor = 0.1):
        self.phi += np.clip(d_phi, -2*np.pi/50, 2*np.pi/50)
        
        return self._get_coords()

    def _get_coords(self) -> list:
        x = np.cos(self.phi) * self.radius
        y = np.sin(self.phi) * self.radius

        return [self.center[0] + x, self.center[1] + y, self.z]

class CameraRailRobotStraight:
    """
    Class is useful to describe the movement of the camera on a straight rail
    """

    def __init__(self, center: Union[List, np.ndarray], center_offset_direction : Union[List, np.ndarray], center_offset_distance : float, length : float, z_height : float = 1.0, max_vel : float= 0.069):
        if type(center) is list: center = np.array(center)
        self.center = center
        if type(center_offset_direction) is list: center_offset_direction = np.array(center_offset_direction)
        self.codir : np.ndarray = center_offset_direction
        self.codir = self.codir/np.linalg.norm(self.codir)

        self.codist = center_offset_distance
        self.length = length
        self.pos_rel_to_length = 0
        self.max_vel = max_vel

        self.vel = 0
        self.position : np.ndarray = center + self.codist * self.codir
        self.position[2] = z_height
        self.vec = np.array([self.codir[1], -self.codir[0], 0])
        self.vec = self.vec / np.linalg.norm(self.vec)

    def _get_coords(self):
        self.vel = np.clip(self.vel, -self.max_vel, self.max_vel)

        self.pos_rel_to_length += self.vel
        if np.abs(self.pos_rel_to_length) > self.length/2:
            self.vel = 0
            self.pos_rel_to_length = np.sign(self.pos_rel_to_length) * self.length/2

        self.position += self.vel * self.vec


    def get_coords(self, d_vel):
        self.vel += d_vel
        self._get_coords()

        return self.position.tolist()

class CameraRailRobotCircle:
    """
    Useful to describe the movement of the camera on a circular path with limitations
    """
    
    def __init__(self, center, radius, z_height, phi_min= -np.pi, phi_max= np.pi, phi_offset = 0, x_y_offset= None, max_step_size= np.pi/25, phi= 0): # phi in RAD
        if type(center) is list: center = np.array(center)
        self.center = center
        self.radius = radius
        self.z = z_height
        self.phi = phi
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.phi_offset = phi_offset
        self.max_step_size = max_step_size
        self.x_y_offset = x_y_offset
        if self.x_y_offset is None:
            self.x_y_offset = [0, 0]
        self.position = self._get_coords()
        self.vel = 0


    def _clip_phi(self):
        self.phi = np.clip(self.phi, self.phi_min + self.phi_offset, self.phi_max + self.phi_offset)

    def get_coords(self, d_phi):
        self.phi += np.clip(d_phi, -self.max_step_size, self.max_step_size)
        
        return self._get_coords()

    def _get_coords(self) -> list:
        self._clip_phi()
        x = np.cos(self.phi) * self.radius
        y = np.sin(self.phi) * self.radius

        return [self.center[0] + x + self.x_y_offset[0], self.center[1] + y + self.x_y_offset[1], self.z]





def add_list(a: List, b: List, factor: int = 1) -> List:
    """
    adds lists "a" and "b" as vectors, "factor" is multiplied by b
    """
    return (np.array(a) + factor * np.array(b)).tolist()

def directionalVectorsFromQuaternion(quaternion: Union[List, torch.Tensor], scale= 1) -> Union[List, torch.Tensor]:
    """
    Returns (scaled) up/forward/left vectors for world rotation frame quaternion.
    """
    x, y, z, w = quaternion
    up_vector = [
        scale*(2* (x*y - w*z)),
        scale*(1- 2* (x*x + z*z)),
        scale*(2* (y*z + w*x)),
    ]
    forward_vector = [
        scale*(2* (x*z + w*y)),
        scale*(2* (y*z - w*x)),
        scale*(1- 2* (x*x + y*y)),
    ]
    left_vector = [
        scale*(1- 2* (x*x + y*y)),
        scale*(2* (x*y + w*z)),
        scale*(2* (x*z - w*y)),
    ]

    if type(quaternion) is torch.Tensor:
        up_vector = torch.tensor(up_vector)
        forward_vector = torch.tensor(forward_vector)
        left_vector = torch.tensor(left_vector)

    return up_vector, forward_vector, left_vector

def add_list(a: List, b: List, factor: int = 1) -> List:
    """
    adds lists "a" and "b" as vectors, "factor" is multiplied by b
    """
    return (np.array(a) + factor * np.array(b)).tolist()

def getOrientationFromDirectionalVector(v: Union[List, np.ndarray], v_base = None) -> List:

    """

    Gets world space quaternion orientation for a directional vector v


    :param v: Directional vector to extract world space orientation
    """
    if type(v) is list:
        v = np.array(v)
    v = v/np.linalg.norm(v)

    if v_base is None:
        v_base = np.array([0,0,1])
    if type(v_base) is list:
        v_base = np.array(v_base)
    v_base = v_base/np.linalg.norm(v_base)

    if np.dot(v_base, v) > 0.999999: return [0, 0, 0, 1]
    if np.dot(v_base, v) < -0.999999: return [0, 0, 0 ,-1]

    a = np.cross(v_base, v)
    q = a.tolist()
    q.append(np.sqrt((np.linalg.norm(v_base, 2)**2) * (np.linalg.norm(v, 2)**2)) + np.dot(v_base, v))
    q = q/np.linalg.norm(q, 2)

    return q

def directionalVectorsFromQuaternion(quaternion: Union[List, torch.Tensor], scale= 1) -> Union[List, torch.Tensor]:
    """
    Returns (scaled) up/forward/left vectors for world rotation frame quaternion.
    """
    x, y, z, w = quaternion
    up_vector = [
        scale*(2* (x*y - w*z)),
        scale*(1- 2* (x*x + z*z)),
        scale*(2* (y*z + w*x)),
    ]
    forward_vector = [
        scale*(2* (x*z + w*y)),
        scale*(2* (y*z - w*x)),
        scale*(1- 2* (x*x + y*y)),
    ]
    left_vector = [
        scale*(1- 2* (x*x + y*y)),
        scale*(2* (x*y + w*z)),
        scale*(2* (x*z - w*y)),
    ]

    if type(quaternion) is torch.Tensor:
        up_vector = torch.tensor(up_vector)
        forward_vector = torch.tensor(forward_vector)
        left_vector = torch.tensor(left_vector)

    return up_vector, forward_vector, left_vector

