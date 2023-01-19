# -*- coding: utf-8 -*-

import numpy as np


class Controller:
    """Base Class for Controllers

    Parameters
    ----------
    timestep : float, optional
        timestep of controller, by default 0.01
    bumper_r : float, optional
        radius of bumper, by default 0.33
    bumper_l : float, optional
        location of bumper from COM, by default 0.2425
    """
    def __init__(self,
                 v_max=np.inf,
                 omega_max=np.inf,
                 timestep=0.01,
                 bumper_r=0.33,
                 bumper_l=0.2425):
        self.v_max = v_max
        self.omega_max = omega_max
        self.timestep = timestep
        self.bumper_l = bumper_l
        self.bumper_r = bumper_r

    def update(self, F, v_prev, omega_prev, v_cmd, omega_cmd):
        """Get updated desired velocity with compliance in mind.

        Parameters
        ----------
        F : ndarray
            Array containing contact forces and moments. Order of force and moments is [Fx, Fy, Fz, Mx, My, Mz]
        v_prev : float
            Previous linear velocity
        omega_prev : float
            Previous angular velocity
        v_cmd : float
            Command linear velocity
        omega_cmd : float
            Command angular velocity

        Returns
        -------
        tuple(float, float)
            Tuple containing linear and rotational velocity after compliant control
        """
        raise NotImplementedError

    def get_location_on_bumper(self, Fx, Fy, Mz):
        """Get collision force and location on bumper.

        Parameters
        ----------
        Fx : float
            Force in X-direction (Towards left)
        Fy : float
            Force in Y-direction (Towards back)
        Mz : float
            Moment in Z-direction

        Returns
        -------
        tuple(float, float, float)
            Tuple containing
                - Fmag, Magnitude of collision force perpendicular to bumper
                - h, Height of the collision point
                - theta, Angle on bumper of the collision point
        """
        self._h = 0

        (a, b, c) = (Fx, Fy, Mz/self.bumper_r)
        temp = a**2 + b**2 - c**2
        if temp > 0:
            self._theta = np.real(-1j * np.log(
                (c + 1j*np.sqrt(temp))
                / (a + 1j*b)
            ))
        else:
            self._theta = np.real(-1j * np.log(
                (c - np.sqrt(-temp))
                / (a + 1j*b)
            ))
        self._Fmag = Fx*np.sin(self._theta) + Fy*np.cos(self._theta)

        return (self._Fmag, self._h, self._theta)
