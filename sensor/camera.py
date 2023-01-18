import torch
import numpy as np
import pybullet as pyb
from gym import spaces
from typing import Union, List, Dict, TypedDict
from sensor.sensor import Sensor
import copy
from time import time
from abc import abstractmethod
from robot.ur5 import UR5

class CameraArgs(TypedDict, total= False):
    width : int
    height : int
    type : str
    up_vector : List[float]
    fov : float
    aspect : float
    near_val : float
    far_val : float

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

class Camera(Sensor):
    """
    This class implements a camera that can be used as any combination of [R,G,B,D] including grayscale.


    param: debug: dict with debug parameters
    """

    def __init__(self, position: List = None, target: List = None, camera_args: CameraArgs = None, orientation: List = None,\
            debug : Dict[str, bool] = None, name : str = 'default', \
            normalize: bool = False, add_to_observation_space: bool = True, add_to_logging: bool = False, sim_step: float = 0):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step)
        self.pos = position if position is not None else [0,0,0]
        self.target = target if target is not None else [0,0,0]
        self.camera_args : CameraArgs
        self._parse_camera_args(camera_args)
        self.orn = [0,0,0,1] if orientation is None else orientation
        self.name = name
        self.output_name = f'camera_{self.camera_args["type"]}_{self.name}'
        self.debug = {
            'position' : False,
            'target' : False,
            'orientation' : False,
            'lines' : False,
        } if debug is None else debug
        self._add_debug_params()

        self.camera = self._set_camera()

    def _parse_camera_args(self, camera_args : CameraArgs):
        default_camera_args : CameraArgs = {
            'width' : 128,
            'height' : 128,
            'type' : 'rgb',
            'up_vector' : [0,0,1],
            'fov' : 60,
            'aspect' : 1,
            'near_val' : 0.05,
            'far_val' : 5,
        }
        self.camera_args = default_camera_args
        if type(camera_args) is dict:
            self._modify_camera_args(camera_args)


    def _add_debug_params(self):
        if self.debug.get('lines', False):
            self.debug_lines = {}
        if self.debug.get('target', False):
            self.target_debug =[pyb.addUserDebugParameter(f'x_target', -5, 5, 0), pyb.addUserDebugParameter(f'y_target', -5, 5, 0.4), pyb.addUserDebugParameter(f'z_target', -2, 2, 0.3)]
        if self.debug.get('orientation', False):
            self.orientation_debug = [pyb.addUserDebugParameter(f'roll', -2*np.pi, 2*np.pi, 0), pyb.addUserDebugParameter(f'pitch', -2*np.pi, 2*np.pi, 0), pyb.addUserDebugParameter(f'yaw', -2*np.pi, 2*np.pi, 0)]
        if self.debug.get('position', False):
            self.position_debug = [pyb.addUserDebugParameter(f'x', -5, 5, 0.7), pyb.addUserDebugParameter(f'y', -5, 5, 0.7), pyb.addUserDebugParameter(f'z', -2, 2, 0.4)]

    def _use_debug_params(self):
        if self.debug.get('position', False):
            x = pyb.readUserDebugParameter(self.position_debug[0])
            y = pyb.readUserDebugParameter(self.position_debug[1])
            z = pyb.readUserDebugParameter(self.position_debug[2])
            self.pos = [x, y, z]

        if self.debug.get('orientation', False):
            roll = pyb.readUserDebugParameter(self.orientation_debug[0])
            pitch = pyb.readUserDebugParameter(self.orientation_debug[1])
            yaw = pyb.readUserDebugParameter(self.orientation_debug[2])
            self.orn = pyb.getQuaternionFromEuler([roll, pitch, yaw])

        if self.debug.get('target', False):
            x = pyb.readUserDebugParameter(self.target_debug[0])
            y = pyb.readUserDebugParameter(self.target_debug[1])
            z = pyb.readUserDebugParameter(self.target_debug[2])
            self.target = [x, y, z]

        if self.debug.get('lines', False):
            for key, line_id in self.debug_lines.items():
                pyb.removeUserDebugItem(line_id)
            self.debug_lines = {}
            self.debug_lines['target'] = pyb.addUserDebugLine(self.pos, self.target, [127, 127, 127])
            # up_vector, forward_vector, left_vector = directionalVectorsFromQuaternion(orientation)
            # self.debug_lines['forward'] = p.addUserDebugLine(self.pos, add_list(self.pos, forward_vector), [255, 0, 0])
            # self.debug_lines['left'] = p.addUserDebugLine(self.pos, add_list(self.pos, left_vector), [0, 255, 0])
            # self.debug_lines['up'] = p.addUserDebugLine(self.pos, add_list(self.pos, up_vector), [0,0,255])

    def _set_camera(self):
        if self.debug.get('position', False) or self.debug.get('orientation', False) or self.debug.get('target', False) or self.debug.get('lines', False):
            self._use_debug_params()

        viewMatrix = pyb.computeViewMatrix(
            cameraTargetPosition=self.target,
            cameraEyePosition= self.pos,
            cameraUpVector= self.camera_args['up_vector'],
            )

        projectionMatrix = pyb.computeProjectionMatrixFOV(
            fov= self.camera_args['fov'],
            aspect=self.camera_args['aspect'],
            nearVal= self.camera_args['near_val'],
            farVal= self.camera_args['far_val'],
            )


        def _set_camera_inner(): # TODO           
            _, _, rgba, depth, _ = pyb.getCameraImage(
                width= self.camera_args['width'],
                height= self.camera_args['height'],
                viewMatrix= viewMatrix,
                projectionMatrix= projectionMatrix,
            )
            if self.camera_args['type'] == 'grayscale':
                r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]/255
                image = (0.2989 * r + 0.5870 * g + 0.1140 * b)*a
                image = image[None]
            if self.camera_args['type'] in ['rgb']:
                image = rgba.copy()[:, :, :3]
            if self.camera_args['type'] == 'rgbd':
                image = rgba.copy()
                image[:, :, 3] = depth


            return image

        self.camera_ready = True
        return _set_camera_inner

    def _get_image(self):
        if not self.camera_ready:
            self.camera = self._set_camera()
        return self.camera() 

    def _move(self, position = None, orientation = None, target = None):
        self.pos = self.pos if position is None else position 
        self.orn = self.orn if orientation is None else orientation 
        self.pos = self.target if target is None else target
        self.camera = self._set_camera()

    def _modify_camera_args(self, new_camera_args: CameraArgs = None) -> CameraArgs:
        """
        Pass the arguments to be modified, the rest will remain the same.
        """
        for key in self.camera_args:
            new_arg = new_camera_args.get(key, None)
            if new_arg is not None:
                assert type(new_arg) == type(self.camera_args[key]), f'Old type <{type(self.camera_args[key])}> is different from new type <{type(new_arg)}>'
                self.camera_args[key] = new_arg
        return copy.copy(self.camera_args)

    def get_observation_space_element(self) -> Dict:
        nr_channels = {
            'grayscale' : 1,
            'rgb' : 3,
            'rgbd': 4,
        }
        return {self.output_name : spaces.Box(low=0, high= 255, shape=(128,128,nr_channels[self.camera_args['type']],), dtype=np.uint8),}
        

    def get_observation(self):
        return {self.output_name : self._get_image()}

    def update(self):
        self.cpu_epoch = time()
        self._adapt_to_environment()
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        self._adapt_to_environment()
        self.cpu_time = time() - self.cpu_epoch

    def _normalize(self):
        pass

    @abstractmethod
    def _adapt_to_environment(self):
        """
        The idea within this method is to change the position/orientation/target of the camera. For example if it is attached to a robot's effector that self.pos = robot.effector_pos.
        Intrinsic camera params can also be changed, but careful not to break something.
        """
        self.camera = self._set_camera()


class StaticBodyCameraUR5(Camera):

    def __init__(self, robot : UR5, position_relative_to_effector: List = None, camera_args: CameraArgs = None, name : str = 'default_body_ur5', **kwargs):
        self.robot = robot
        self.relative_pos = position_relative_to_effector
        super().__init__(camera_args= camera_args, name= name, **kwargs)

    def _calculate_position(self):
        effector_position, effector_orientation = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4:6]
        body_position, body_orientation = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id - 1)[4:6]
        effector_up_vector, effector_forward_vector, _ = directionalVectorsFromQuaternion(effector_orientation)
        self.camera_args['up_vector'] = effector_up_vector
        if self.relative_pos is None:
            target = add_list(effector_position, effector_forward_vector) # [p+v for p,v in zip(effector_position, effector_forward_vector)]
            body_forward_vector, body_up_vector, _ = directionalVectorsFromQuaternion(body_orientation)
            position = add_list(add_list(body_position, body_up_vector, 0.075), body_forward_vector, 0.075) # [p+u+f for p,u,f in zip(body_position, body_up_vector, body_forward_vector)]
        else:
            position = add_list(effector_position, self.relative_pos)
            target = add_list(position, effector_forward_vector)
        
        return position, target

    def _adapt_to_environment(self):
        self.pos, self.target = self._calculate_position()
        super()._adapt_to_environment()


class StaticFloatingCamera(Camera):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, robot : UR5, position: List, target: List = None, camera_args: CameraArgs = None, name : str = 'default_floating', **kwargs):
        super().__init__(target= target, camera_args= camera_args, name= name, **kwargs)
        self.robot = robot
        self.follow_effector = target is None
        self.pos = position

    def _adapt_to_environment(self):
        if self.follow_effector:
            self.target = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4]
        super()._adapt_to_environment()






















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

