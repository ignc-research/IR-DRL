from .camera import CameraBase
from .camera_implementations import *


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


CameraRegistry.register('UR5_Bodycam')(StaticBodyCameraUR5)
CameraRegistry.register('General_Floating')(StaticFloatingCamera)

