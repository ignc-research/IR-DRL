from .camera import CameraBase
from .camera_implementations.static_cameras import *
from .camera_implementations.on_robot_cameras import *


class CameraRegistry:
    _camera_classes = {}

    @classmethod
    def get(cls, camera_type:str) -> CameraBase:
        try:
            return cls._camera_classes[camera_type]
        except KeyError:
            raise ValueError(f"unknown camera type : {camera_type}")

    @classmethod
    def register(cls, camera_type:str):
        def inner_wrapper(wrapped_class):
            cls._camera_classes[camera_type] = wrapped_class
            return wrapped_class
        return inner_wrapper


CameraRegistry.register('OnBody_UR5')(OnBodyCameraUR5)
CameraRegistry.register('Floating_General')(StaticFloatingCamera)
CameraRegistry.register('Floating_FollowEffector')(StaticFloatingCameraFollowEffector)

