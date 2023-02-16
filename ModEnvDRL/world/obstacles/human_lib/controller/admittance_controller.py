# -*- coding: utf-8 -*-

import numpy as np

from . import Controller


class AdmittanceController(Controller):
    """Admittance ControllerS

    Attributes
    ----------
    damping_gain : float
        Damping gain to use
    robot_mass : float
        Maximum effective mass of the robot
    collision_F_max : float
        Maximum allowed collision force
    activation_F : float
        Activation collision force

    Notes
    -----
    Implements compliant controller based on following control equation with reference set to 0
    and discretised with ZOH.
    .. math:: F = robot_mass \Delta \ddot{x} + Damping_gain \Delta \dot{x} + K \Delta x

    """

    def __init__(self,
                 damping_gain=0.2,
                 robot_mass=5,
                 collision_F_max=45,
                 activation_F=15,
                 **kwargs):
        super().__init__(**kwargs)
        self.damping_gain = damping_gain
        self.robot_mass = robot_mass
        self.collision_F_max = collision_F_max
        self.activation_F = activation_F
        self.nominal_F = self.collision_F_max

        # Internal State Variables
        self._Fmag = 0.0
        self._theta = 0.0
        self._h = 0.0
        self._p = 0.0
        self._Fx = 0.0
        self._Fy = 0.0
        self._Mz = 0.0

        self.V_contact = 0.0

    def update(self, F, v_prev, omega_prev, v_cmd, omega_cmd):
        (self._Fx, self._Fy, self._Mz) = (F[0], F[1], F[5])
        self.get_location_on_bumper(self._Fx, self._Fy, self._Mz)
        if self._Fmag > self.activation_F:
            return self.__control(v_prev, omega_prev, v_cmd, omega_cmd)
        else:
            self.V_contact = np.nan
            self._theta = np.nan
            return (v_cmd, omega_cmd)

    def __control(self, v_prev, omega_prev, v_cmd, omega_cmd):
        """Get new velocity

        Parameters
        ----------
        v_prev : float
            Prev linear velocity
        omega_prev : float
            Prev rotational velocity
        v_prev : float
            Demand linear velocity
        omega_prev : float
            Demand rotational velocity

        Returns
        -------
        tuple(float, float)
            Tuple containing linear and rotational velocity after compliant control
        """
        stheta = np.sin(self._theta)    # Small optimization
        ctheta = np.cos(self._theta)    # Small optimization

        # Position wrt center of rotatiion
        O = np.sqrt((self.bumper_r*stheta)**2
                    + (self.bumper_l + self.bumper_r*ctheta)**2)
        beta = np.arctan2(self.bumper_r * stheta, self.bumper_l + self.bumper_r*ctheta)

        sbeta = np.sin(beta)      # Small optimization
        cbeta = np.cos(beta)      # Small optimization

        # Admittance Control
        a = ctheta
        b = O*(stheta*cbeta - ctheta*sbeta)

        V_prev = (a * v_prev) + (b * omega_prev)
        V_cmd = (a * v_cmd) + (b * omega_cmd)

        eff_robot_mass = self.robot_mass
        if (abs(V_cmd) > (self.collision_F_max * self.timestep) / self.robot_mass):
            eff_robot_mass = (self.collision_F_max * self.timestep) / abs(V_cmd)

        V_dot = (self.nominal_F - self._Fmag - self.damping_gain*V_prev) / eff_robot_mass
        V = V_dot * self.timestep + V_cmd
        self.V_contact = V

        # Calculate new v and omega in parameterized form
        a = 1.0
        b = -(stheta*cbeta - ctheta*sbeta) / self.omega_max * self.v_max

        # Ensure non-zero 'a' and 'b'
        eps = 0.01
        if (abs(a) < eps):
            return (v_cmd, V/b)
        if (abs(b) < eps):
            return (V/a, omega_cmd)

        _ = V - a*v_cmd / b
        if _ > self.omega_max:
            t_max = (self.omega_max - omega_cmd) / (_ - omega_cmd)
        elif _ < -self.omega_max:
            t_max = (-self.omega_max - omega_cmd) / (_ - omega_cmd)
        else:
            t_max = 1.0

        _ = V - b*omega_cmd / a
        if _ > self.v_max:
            t_min = (self.v_max - omega_cmd) / (_ - omega_cmd)
        elif _ < -self.v_max:
            t_min = (-self.v_max - omega_cmd) / (_ - omega_cmd)
        else:
            t_min = 0.0

        self._p = self.__map(np.abs(self._theta),
                             [0.0, np.pi],
                             [t_min, t_max])

        v = self._p * v_cmd + (1-self._p) * (V - b*omega_cmd) / a
        omega = self._p * (V - a*v_cmd) / b + (1-self._p) * omega_cmd

        return (v, omega)

    def __map(self, x, from_range, to_range):
        return (to_range[0]
                + ((x - from_range[0])
                   * (to_range[1] - to_range[0])
                   / (from_range[1] - from_range[0])))
