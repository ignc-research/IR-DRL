import numpy as np
import pybullet as pyb
from gym import spaces
from typing import Union, List, Dict, TypedDict
from ModEnvDRL.sensor.sensor import Sensor
import copy
from time import time
from abc import abstractmethod
from .camera_utils import *

class CameraArgs(TypedDict, total= False):
    width : int
    height : int
    type : str
    up_vector : List[float]
    fov : float
    aspect : float
    near_val : float
    far_val : float


class CameraBase(Sensor):
    """
    This class implements a camera that can be used as any combination of [R,G,B,D] including grayscale.


    param: debug: dict with debug parameters
    """

    def __init__(self, position: List = None, target: List = None, camera_args: CameraArgs = None, orientation: List = None,\
            debug : Dict[str, bool] = None, name : str = 'default', \
            normalize: bool = False, add_to_observation_space: bool = True, add_to_logging: bool = False, sim_step: float = 0, update_steps: int = 0):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, update_steps)
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

        self.current_image = None

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
            rgba, depth = np.array(rgba), np.array(depth) # for compatibility with older python versions
            if self.camera_args['type'] == 'grayscale':
                r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]/255
                image = (0.2989 * r + 0.5870 * g + 0.1140 * b)*a
                image = image[None]
            if self.camera_args['type'] in ['rgb']:
                image = rgba[:, :, :3]
            if self.camera_args['type'] == 'rgbd':
                image = rgba
                image[:, :, 3] = depth


            return image

        self.camera_ready = True
        return _set_camera_inner

    def _get_image(self):
        if not self.camera_ready:
            self.camera = self._set_camera()
        self.image = self.camera()
        return self.image/255 if self.normalize else self.image 

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
        low = 0
        high = 1 if self.normalize else 255
        return {self.output_name : spaces.Box(low=low, high= high, shape=(128,128,nr_channels[self.camera_args['type']],), dtype=np.uint8),}
        

    def get_observation(self):
        return {self.output_name : self.current_image}

    def update(self, step):
        self.cpu_epoch = time()
        if step % self.update_steps == 0:
            self._adapt_to_environment()
            self.current_image = self._get_image()
        self.cpu_time = time() - self.cpu_epoch

        return self.get_observation()

    def reset(self):
        self.cpu_epoch = time()
        self._adapt_to_environment()
        self.cpu_time = time() - self.cpu_epoch

    def _normalize(self):
        """
        Bin mir immer noch nicht sicher wie das hier funktionieren soll
        """
        pass

    def get_data_for_logging(self) -> dict:
        """
        
        """
        if not self.add_to_logging:
            return {}
        logging_dict = dict()

        logging_dict[self.output_name + '_cpu_time'] = self.cpu_time

        return logging_dict

    @abstractmethod
    def _adapt_to_environment(self):
        """
        The idea within this method is to change the position/orientation/target of the camera. For example if it is attached to a robot's effector that self.pos = robot.effector_pos.
        Intrinsic camera params can also be changed, but careful not to break something.
        """
        self.camera = self._set_camera()

