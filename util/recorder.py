import pybullet as pyb
from typing import Union
from robot.robot import Robot
from world.obstacles.pybullet_shapes import Box, Sphere, Cylinder
from world.obstacles.urdf_object import URDFObject
from pickle import dumps

__all__ = [
    "Recorder"
]

class RoboTracker:
    """
    Small class to make tracking all of a robot's links easy.
    """

    def __init__(self, robot: Robot):
        self.urdf_path = robot.urdf_path
        self.object_id = robot.object_id
        joints = [pyb.getJointInfo(self.object_id, i) for i in range(pyb.getNumJoints(self.object_id))]
        self.links = [jointInfo[16] for jointInfo in joints if jointInfo[16] != -1]  # ignore base link, we already cover that elsewhere

    def _process(self):
        out = dict()
        out["type"] = "Robot"
        out["config"] = dict()
        out["config"]["urdf_path"] = self.urdf_path
        out["config"]["links"] = dict()
        for linkID in self.links:
            link_info = pyb.getLinkState(self.object_id, linkID, computeForwardKinematics=True)
            out["config"]["links"][linkID] = {"pos": link_info[4], "quat": link_info[5]}
        return out

class Recorder:
    """
    Class to record the movements of objects in the PyBullet simulation.
    The data can be dumped and later be used for creating renderings in tools like Blender.
    """

    def __init__(self):
        self.frames = []
        self.tracked_objects = []

    def register_object(self, object: Union[Robot, Box, Sphere, Cylinder, URDFObject]):
        if issubclass(object, Robot):  
            robo_tracker = RoboTracker(object)
            self.tracked_objects.append(robo_tracker)
        else:
            self.tracked_objects.append(object)

    def save_frame(self):
        frame = []
        for tracked_object in self.tracked_objects:   
            id = tracked_object.object_id
            base_pos, base_quat = pyb.getBasePositionAndOrientation(id)
            obj_info = dict()
            obj_info["base_pos"] = base_pos
            obj_info["base_quat"] = base_quat
            if type(tracked_object) is RoboTracker:
                info = tracked_object._process()
            elif type(tracked_object) is Box:
                info = self._process_box(tracked_object)
            elif type(tracked_object) is Cylinder:
                info = self._process_cylinder(tracked_object)
            elif type(tracked_object) is Sphere:
                info = self._process_sphere(tracked_object)
            elif type(tracked_object) is URDFObject:
                info = self._process_urdfobject(tracked_object)
            obj_info = {**obj_info, **info}
            frame.append(obj_info)
        self.frames.append(frame)

    def _process_box(self, box: Box):
        out = dict()
        out["type"] = "Box"
        out["config"] = dict()
        out["config"]["color"] = box.color
        out["config"]["dims"] = box.halfExtents
        return out

    def _process_cylinder(self, cylinder: Cylinder):
        out = dict()
        out["type"] = "Cylinder"
        out["config"] = dict()
        out["config"]["color"] = cylinder.color
        out["config"]["radius"] = cylinder.radius
        out["config"]["height"] = cylinder.height
        return out

    def _process_sphere(self, sphere: Sphere):
        out = dict()
        out["type"] = "Sphere"
        out["config"] = dict()
        out["config"]["color"] = sphere.color
        out["config"]["radius"] = sphere.radius
        return out

    def _process_urdfobject(self, urdfobject: URDFObject):
        out = dict()
        out["type"] = "URDFObject"
        out["config"] = dict()
        out["config"]["scale"] = urdfobject.scale
        out["config"]["urdf_path"] = urdfobject.urdf_path
        return out

    def reset(self):
        self.frames = []
        self.tracked_objects = []

    def save_record(self, file_path):
        pickled_frames = dumps(self.frames)
        with open(file_path, "wb") as outfile:
            outfile.write(pickled_frames)