import os
import math

import pybullet as p
import numpy as np

from .. import Robot


class Wheelchair(Robot):
    """Class for Wheelchair robot.

    Parameters
    ----------
    pybtPhysicsClient : int
        Handle for PyBullet's client
    fixedBase : bool, optional
        Flag to have static robot, by default True
    self_collisions : bool, optional
        Flag to activate self-collisions in the robot, by default False
    v : int, optional
        Initial linear velocity, by default 0
    omega : int, optional
        Initial angular velocity, by default 0
    timestep : float, optional
        Timestep for the robot simulation, by default 0.01
    scaling : float, optional
        Scaling for robot size, by default 1.0

    Attributes
    ----------
    V_MAX : float
        Maximum linear velocity of the robot
    OMEGA_MAX : float
        Maximum angular velocity of the robot
    body_id : int
        Body ID in PyBullet client
    scaling : float
        Scaling for robot size
    global_xyz : ndarray
        Array for global location
    global_quaternion : ndarray
        Array for global orientation
    yaw_angle : float
        Global yaw angle of the robot
    wheel_phase : ndarray
        Array containing wheel's angular location for visualisation
    v : float
        Linear velocity of the robot
    omega : float
        Angular velocity of the robot
    wheel_speed : float
        Array containing wheel's angular velocity for visualisation
    timestep : float
        Timestep for simualtion
    """
    # Robot limitation on speed
    V_MAX = 1.5
    OMEGA_MAX = 1.0

    # Bumper Specs
    ftsensor_loc = [0.35, 0.0]
    bumper_r = 0.33
    bumper_l = 0.2425

    def __init__(
        self,
        pybtPhysicsClient,
        fixedBase=True,
        self_collisions=False,
        with_rider=True,
        v=0,
        omega=0,
        v_max=1.5,
        omega_max=1.0,
        timestep=0.01,
        scaling=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        if with_rider:
            urdf_file = "wheelchair_with_rider.urdf"
        else:
            urdf_file = "wheelchair.urdf"
        self.body_id = p.loadURDF(
            os.path.join(os.path.dirname(__file__), urdf_file),
            flags=p.URDF_MAINTAIN_LINK_ORDER,
            physicsClientId=pybtPhysicsClient,
            globalScaling=scaling,
            useFixedBase=fixedBase,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, -math.pi/2])
        )
        self.scaling = scaling
        self.set_color()

        # Pose
        self.reset()

        # Robot Speed
        self.set_speed(v, omega)

        # Max. Robot Speed
        self.v_max = v_max
        self.omega_max = omega_max

        # Time Step
        self.timestep = timestep

    def set_speed(self, v, omega):
        """Set robot speed

        Parameters
        ----------
        v : float
            Linear velocity
        omega : float
            Angular velocity
        """
        self.v = np.clip(v, -self.V_MAX, self.V_MAX)
        self.omega = np.clip(omega, -self.OMEGA_MAX, self.OMEGA_MAX)

    def set_color(self):
        """Set color to the robot"""
        sdl = p.getVisualShapeData(self.body_id)
        colors = [
            [0.4, 0.4, 0.4, 1],	 # Main Body
            [0.4, 0.4, 0.4, 1],	 # Bumper
            [0.9, 0.8, 0.7, 1],	 # Rider
            [0.9, 0.8, 0.7, 1],	 # Rider
        ]
        for i in range(len(sdl)):
            try:
                p.changeVisualShape(self.body_id, sdl[i][1], rgbaColor=colors[i])
            except Exception:
                pass

    def advance(self):
        """Step simulation by a single timestep"""
        self.global_xyz += self.timestep * np.array([self.v * np.sin(self.yaw_angle),
                                                     -self.v * np.cos(self.yaw_angle),
                                                     0])
        self.yaw_angle += self.timestep * self.omega
        self.global_quaternion = p.getQuaternionFromAxisAngle((0, 0, 1), self.yaw_angle)

    def reset(self):
        """Reset robot to origin for new simulation"""
        self.global_xyz = np.zeros(3)
        self.global_rpy = np.zeros(3)
        self.wheel_phase = np.zeros(2)
        self.yaw_angle = 0
