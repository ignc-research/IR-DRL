"""Collision Module.

This module implements collisions
"""

__all__ = ['Collision']
__version__ = '0.1'
__author__ = 'Vaibhav Gupta'

import numpy as np
import pybullet as p

# Based on ISO/TS 15066 for 75 kg
eff_mass_human = np.array([
    40, 	   # chest
    40, 	   # belly
    40, 	   # pelvis
    75, 75,    # upper legs    <-- Previous implementation was different
    75, 75,    # shins         <-- Previous implementation was different
    75, 75,    # ankles/feet (Same as shin)    <-- Previous implementation was different
    3, 3,      # upper arms
    2, 2,      # forearms
    0.6, 0.6,  # hands
    1.2, 	   # neck
    8.8, 	   # head + face (?)
    75, 75,    # soles (Same as shin)  <-- Previous implementation was different
    75, 75,    # toes (Same as shin)   <-- Previous implementation was different
    40, 	   # chest (back)
    40, 	   # belly (back)
    40, 	   # pelvis (back)
    1.2, 	   # neck (back)
    4.4, 	   # head (back)
]) / 75


eff_spring_const_human = np.array([
    25, 	 # chest
    10, 	 # belly
    25, 	 # pelvis
    50, 50,  # upper legs
    60, 60,  # shins
    75, 75,  # ankles/feet (Same as shin)    <-- From Previous implementation
    30, 30,  # upper arms    <-- Previous implementation was different
    40, 40,  # forearms      <-- Previous implementation was different
    75, 75,  # hands
    50, 	 # neck
    150, 	 # head (face ?)
    75, 75,  # soles (Same as shin)  <-- From Previous implementation
    75, 75,  # toes (Same as shin)   <-- From Previous implementation
    35, 	 # chest (back)
    35, 	 # belly (back)
    35, 	 # pelvis (back)
    50, 	 # neck (back)
    150, 	 # head (back)
]) * 1e3


eff_mass_robot = np.array([
    50,   # Main Body
    1.5,  # Left Wheel
    1.5,  # Right Wheel
    1.5,  # Bumper
]) / 54.5


eff_spring_const_robot = np.array([
    10,  # Main Body
    1,   # Left Wheel
    1,   # Right Wheel
    10,  # Bumper
]) * 1e3


class Collision:
    """[summary]

    Parameters
    ----------
    pybtPhysicsClient : int
        Handle for PyBullet's client
    robot : Robot
        Robot object
    human : Human
        Human object
    human_mass : float, optional
        Weight of human being, by default 75.0
    robot_mass : float, optional
        Weight of the robot, by default 54.5
    bumper_height : float, optional
        Height of the top of the bumper, by default 0.215
    ftsensor_loc : [float, float], optional
        Location of FT sensor from COM, by default [0.035, 0.0]

    Attributes
    ----------
    pybtPhysicsClient : int
        Handle for PyBullet's client
    robot : Robot
        Robot object
    human : Human
        Human object
    human_mass : float
        Weight of human being
    robot_mass : float
        Weight of the robot
    bumper_height : float
        Height of the top of the bumper
    ftsensor_loc : [float, float]
        Location of FT sensor from COM
    eff_mass_robot : float
        Effective mass of the robot for the collision
    eff_mass_human : float
        Effective mass of the human being for the collision
    eff_spring_const : float
        Effective spring constant for the collision
    """

    def __init__(
        self,
        pybtPhysicsClient,
        robot,
        human,
        human_mass=75.0,
        robot_mass=54.5,
        bumper_height=0.215,
        ftsensor_loc=[0.035, 0.0],
        timestep=0.01,
    ):
        self.pybtPhysicsClient = pybtPhysicsClient
        self.robot = robot
        self.human = human
        self.human_mass = human_mass
        self.robot_mass = robot_mass
        self.bumper_height = bumper_height
        self.ftsensor_loc = ftsensor_loc
        self.timestep = timestep

    def get_collision_force(self):
        """Get collision force in case of collision

        Returns
        -------
        None or ndarray
            In case of no collision, returns `None` otherwise returns an array containing contact forces and moments.
            Order of force and moments is [Fx, Fy, Fz, Mx, My, Mz]
        """
        contact_points = p.getContactPoints(
            self.human.body_id,
            self.robot.body_id,
        )

        for contact_point in contact_points:
            if contact_point[8] <= 0:
                # Penetration or Contact
                human_part_id = contact_point[3]
                robot_part_id = contact_point[4]
                pos_on_robot = contact_point[6]

                self.__collide(robot_part_id, human_part_id)
                Fmag = self.__get_contact_force(contact_point[8])
                (h, theta) = self.__get_loc_on_bumper(pos_on_robot)

                self.delta_v, self.delta_omega = self.collision_dynamics(
                    pos_on_robot, Fmag, theta
                )

                return (
                    Fmag * np.sin(theta),
                    Fmag * np.cos(theta),
                    0,
                    -Fmag * np.cos(theta) * h,
                    Fmag * np.sin(theta) * h,
                    0,
                )

        return None

    def collision_dynamics(self, pos, Fmag, theta):
        Vx = Fmag * np.sin(theta) * self.timestep / self.robot_mass
        Vy = Fmag * np.cos(theta) * self.timestep / self.robot_mass
        return self.__cartesian_to_differential(pos, Vx, Vy)

    def __collide(self, robot_part_id, human_part_id):
        """Store parameters based on the colliding parts of the robot and the human

        Parameters
        ----------
        robot_part_id : int
            Part ID of colliding part of robot
        human_part_id : int
            Part ID of colliding part of human
        """
        self.eff_mass_robot = self.__get_eff_mass_robot(robot_part_id)
        self.eff_mass_human = self.__get_eff_mass_human(human_part_id)

        k_robot = self.__get_eff_spring_const_human(robot_part_id)
        k_human = self.__get_eff_spring_const_human(human_part_id)
        self.eff_spring_const = 1 / (1/k_robot + 1/k_human)

    def __get_eff_mass_human(self, part_id):
        """Get effective human mass based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective mass of colliding part of the human
        """
        return eff_mass_human[part_id] * self.human_mass

    def __get_eff_spring_const_human(self, part_id):
        """Get effective spring constant of the human based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective spring constant of colliding part of the human
        """
        return eff_spring_const_human[part_id]

    def __get_eff_spring_const_robot(self, part_id):
        """Get effective spring constant of the robot based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective spring constant of colliding part of the robot
        """
        return eff_spring_const_robot[part_id]

    def __get_eff_mass_robot(self, part_id):
        """Get effective robot mass based on colliding part

        Parameters
        ----------
        part_id : int
            Part ID of colliding part

        Returns
        -------
        float
            Effective mass of colliding part of the robot
        """
        return eff_mass_robot[part_id] * self.robot_mass

    def __get_contact_force(self, penetration):
        """Get contact force based on penetration

        Parameters
        ----------
        penetration : float
            Penetration of robot into human

        Returns
        -------
        float
            Effective Contact Force
        """
        if penetration > 0:
            # No Contact
            return 0
        else:
            return (-self.eff_spring_const * penetration)

    def __get_loc_on_bumper(self, pos):
        theta = np.arctan2(pos[0] - self.ftsensor_loc[1], - pos[1] - self.ftsensor_loc[0])
        h = self.bumper_height - pos[2]
        return (h, theta)

    def __cartesian_to_differential(self, pos, vx, vy):
        pt = (-pos[1], pos[0])
        self.inv_jacobian = np.array([
            [pt[1]/pt[0], 1.],
            [-1./pt[0], 0.],
        ])
        return (self.inv_jacobian @ np.array([vx, vy]))
