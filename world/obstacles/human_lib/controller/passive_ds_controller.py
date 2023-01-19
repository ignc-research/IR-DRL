# -*- coding: utf-8 -*-

import numpy as np

from . import Controller


class PassiveDSController(Controller):
    """Passive DS Controller

    Attributes
    ----------
    damping_gain : float
        Damping gain to use
    robot_mass : float
        Effective mass of the robot.

    Notes
    -----
    Based on Passive DS
    """

    def __init__(self,
                 robot_mass=2,
                 lambda_t=0.0,
                 lambda_n=0.5,
                 Fd=45,
                 activation_F=15,
                 **kwargs):
        super().__init__(**kwargs)
        self.robot_mass = robot_mass
        self.activation_F = activation_F
        self.Lambda = np.diag([lambda_t, lambda_n])
        self.Fd = Fd
        self.D = self.Lambda

        # Internal State Variables
        self._Fmag = 0.0
        self._theta = 0.0
        self._h = 0.0
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
        v_cmd : float
            Demand linear velocity
        omega_cmd : float
            Demand rotational velocity

        Returns
        -------
        tuple(float, float)
            Tuple containing linear and rotational velocity after compliant control
        """
        # Jacobian
        control_pt_x = self.bumper_r * np.cos(self._theta)
        control_pt_y = self.bumper_r * np.sin(self._theta)
        self.jacobian = np.array([
            [1., -control_pt_y],
            [0., control_pt_x],
        ])
        self.inv_jacobian = np.array([
            [1., control_pt_y/control_pt_x],
            [0., 1./control_pt_x],
        ])

        n_hat = np.array([np.cos(self._theta), np.sin(self._theta)])
        t_hat = np.array([-np.sin(self._theta), np.cos(self._theta)])

        Q = np.array([t_hat, n_hat]).T
        self.D = Q @ self.Lambda @ Q.T

        V_prev = np.array(self.__differential_to_cartesian(v_prev, omega_prev))
        V_cmd = np.array(self.__differential_to_cartesian(v_cmd, omega_cmd))

        V = self.timestep / self.robot_mass * (
            - np.matmul(self.D, V_prev)
            + self.Fd * n_hat
            - self._Fmag * n_hat
        ) + np.matmul(t_hat.T, V_cmd) * t_hat

        # Vd = np.matmul(t_hat.T, V_cmd) * t_hat + (self.Fd - self._Fmag)/self.Lambda[1, 1]*n_hat
        # V = self.timestep / self.robot_mass * np.matmul(self.D, (Vd - V_prev))

        self.V_contact = n_hat.T @ V

        return self.__cartesian_to_differential(V[0], V[1])

    def __differential_to_cartesian(self, v, omega):
        return (self.jacobian @ np.array([v, omega]))

    def __cartesian_to_differential(self, vx, vy):
        return (self.inv_jacobian @ np.array([vx, vy]))
