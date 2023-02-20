"""Simulator Module.

This module implements simualtion using PyBullet
"""

__all__ = ['Simulator']
__version__ = '0.1'
__author__ = 'Vaibhav Gupta'

import time
import os
import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pybullet as p
import pybullet_data
import numpy as np

from collision import Collision


def pos_atan(y, x):
    a = np.arctan2(y, x)
    if a < 0.0:
        a += 2*np.pi
    return a


def reset_human(human, distance, robot_angle, human_angle, gait_phase):
    human.reset()
    x = distance*np.cos(-np.pi/2-robot_angle)
    y = distance*np.sin(-np.pi/2-robot_angle)
    orientation = -np.pi/2-robot_angle + human_angle
    human.resetGlobalTransformation(
        xyz=np.array([x, y, 0.94*human.scaling]),
        rpy=np.array([0, 0, orientation-np.pi/2]),
        gait_phase_value=0
    )


class Ground:
    def __init__(self,
                 pybtPhysicsClient,
                 urdf_path=os.path.join(pybullet_data.getDataPath(), "plane.urdf")):
        self.id = p.loadURDF(
            urdf_path,
            physicsClientId=pybtPhysicsClient,
        )

    def advance(self, global_xyz, global_quaternion):
        p.resetBasePositionAndOrientation(
            self.id,
            global_xyz,
            global_quaternion
        )


class Simulator:
    DISTANCE = 2.0

    def __init__(
        self,
        Robot,
        Human,
        Controller,
        walker_scaling=1.0,
        show_GUI=True,
        timestep=0.01,
        collision_timestep=0.01,
        make_video=False,
        fast_forward=False
    ):
        self.timestep = timestep
        self.collision_timestep = collision_timestep
        self.t_max = 20.0 / walker_scaling

        # Objects
        self.Robot = Robot
        self.Human = Human
        self.Controller = Controller

        # define constants for the setup
        distance = self.DISTANCE
        robot_radius = 0.6
        human_radius = 0.6 * walker_scaling
        self.nominal_human_speed = 1.1124367713928223 * 0.95 * walker_scaling
        self.nominal_robot_speed = 1.0

        miss_angle_tmp = np.arccos(np.sqrt(1 - (robot_radius+human_radius)**2 / distance**2))
        self.miss_angle_lower_threshold = np.pi - miss_angle_tmp
        self.miss_angle_upper_threshold = np.pi + miss_angle_tmp
        self.miss_speed_threshold = (distance - human_radius - robot_radius) / self.t_max

        # set up Bullet with the robot and the walking man
        self.show_GUI = show_GUI
        if self.show_GUI:
            self.make_video = make_video
        else:
            self.make_video = False
        self.__setup_world()

    def plot_collision_forces(self,
                              collision_forces,
                              robot_target_velocities,
                              robot_cmd_velocities,
                              contact_pt_velocity,
                              contact_pt_loc):
        f, ax = plt.subplots(2, 2, sharex=True)

        time = np.arange(collision_forces.shape[0]) * self.collision_timestep
        # ax[0, 0].plot(time, np.linalg.norm(collision_forces[:, 0:2], axis=1))
        ax[0, 0].plot(time, collision_forces[:, -1])
        ax[0, 0].set_xlabel("Time [s]")
        ax[0, 0].set_ylabel("Force [N]")

        ax[0, 1].plot(time, robot_target_velocities[:, 0], color=(0, 0.4470, 0.7410, 1))
        ax[0, 1].plot(time, robot_cmd_velocities[:, 0], linestyle="--", color=(0, 0.4470, 0.7410, 0.8))
        ax[0, 1].set_xlabel("Time [s]")
        ax[0, 1].set_ylabel("v [m/s]", color=(0, 0.4470, 0.7410, 1))
        ax[0, 1].tick_params(axis='y', labelcolor=(0, 0.4470, 0.7410, 1))

        ax_ = ax[0, 1].twinx()
        ax_.plot(time, robot_target_velocities[:, 1], color=(0.8500, 0.3250, 0.0980, 1))
        ax_.plot(time, robot_cmd_velocities[:, 1], linestyle="--", color=(0.8500, 0.3250, 0.0980, 0.8))
        ax_.set_ylabel("omega [rad/s]", color=(0.8500, 0.3250, 0.0980, 1))
        ax_.tick_params(axis='y', labelcolor=(0.8500, 0.3250, 0.0980, 1))

        ax[1, 0].plot(time, contact_pt_loc)
        ax[1, 0].set_xlabel("Time [s]")
        ax[1, 0].set_ylabel("theta [deg]")

        if len(contact_pt_velocity) > 0:
            time = np.arange(contact_pt_velocity.shape[0]) * self.collision_timestep
            ax[1, 1].plot(time, contact_pt_velocity)
            ax[1, 1].set_xlabel("Time [s]")
            ax[1, 1].set_ylabel("V_contact [m/s]")

        plt.tight_layout()

        if self.make_video:
            i = 0
            while os.path.exists(os.path.join("media", "plot_{:d}.png".format(i))):
                i += 1
            plt.savefig(os.path.join("media", "plot_{:d}.png".format(i)))
        else:
            plt.show()

    def simulate(
        self,
        robot_angle=0,
        human_angle=0,
        gait_phase=0,
        human_speed_factor=1.0,
        robot_speed_factor=0.6
    ):
        human_speed = self.nominal_human_speed * human_speed_factor
        robot_speed = self.nominal_robot_speed * robot_speed_factor
        human_velocity = human_speed*np.array([np.cos(human_angle), np.sin(human_angle)])
        robot_velocity = robot_speed*np.array([np.cos(robot_angle), np.sin(robot_angle)])
        relative_velocity = human_velocity - robot_velocity
        relative_speed = np.sqrt(np.dot(relative_velocity, relative_velocity))
        angle_relative_v = pos_atan(relative_velocity[1], relative_velocity[0])

        if (
            self.miss_angle_lower_threshold < angle_relative_v < self.miss_angle_upper_threshold
            and relative_speed > self.miss_speed_threshold
        ):
            # Collision is possible
            self.cmd_robot_speed = (robot_speed, 0)
            self.collision_over = False
            self.robot.set_speed(*self.cmd_robot_speed)
            self.robot.reset()

            t = 0
            reset_human(self.human, self.DISTANCE, robot_angle, human_angle, gait_phase)
            collision_forces = []
            robot_target_velocities = []
            robot_cmd_velocities = []
            contact_pt_velocity = []
            contact_pt_loc = []
            sim_timestep = self.timestep
            self.robot.timestep = self.timestep
            self.human.timestep = self.timestep
            self.controller.timestep = self.timestep
            self.collider.timestep = self.timestep
            p.setTimeStep(self.timestep, self.physics_client_id)

            if self.make_video:
                ani_fig = plt.figure(figsize=(32, 18))
                ax = plt.subplot(111)
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.axis('off')
                mat = np.random.random((1080, 1920))
                image = ax.imshow(mat, interpolation='none', animated=True, label="Video")
                ani_fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                i = 0
                while os.path.exists(os.path.join("media", "video_{:d}.mp4".format(i))):
                    i += 1
                metadata = dict(title='Video', artist='Human Robot Collider')
                writer = animation.FFMpegWriter(fps=15, metadata=metadata, bitrate=-1, codec="h264")
                writer.setup(ani_fig, os.path.join("media", "video_{:d}.mp4".format(i)))

            t_collision_over = None
            t_collision_start = None
            while t < self.t_max:
                sim_timestep = self.__step(
                    collision_forces,
                    robot_target_velocities,
                    robot_cmd_velocities,
                    contact_pt_velocity,
                    contact_pt_loc,
                )
                t += sim_timestep

                # Save Video Frame for recording
                if self.make_video:
                    if t % 0.1 < 1e-6 or 0.1 - (t % 0.1) < 1e-6:
                        w, h, img, _, _ = p.getCameraImage(
                            1920, 1080,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                            flags=p.ER_NO_SEGMENTATION_MASK
                        )
                        # img = np.reshape(img, (w, w, 4)) * (1. / 255.)
                        image.set_data(img)
                        writer.grab_frame()

                # # Switch cmd. speed to 0 after 2s of collision
                # if len(collision_forces) > 0:
                #     if t_collision_start is None:
                #         t_collision_start = t
                #     if t - t_collision_start > 2:
                #         self.cmd_robot_speed = (0, 0)

                # Exit simulation 2 sec after collision is over
                if self.collision_over:
                    if t_collision_over is None:
                        t_collision_over = t
                    if t - t_collision_over > 3:
                        break

                # Exit simulation on "q" key press
                keys = p.getKeyboardEvents(self.physics_client_id)
                qKey = ord('q')
                if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                    break

                if self.show_GUI:
                    time.sleep(sim_timestep)

            collision_forces = np.array(collision_forces)
            robot_target_velocities = np.array(robot_target_velocities)
            robot_cmd_velocities = np.array(robot_cmd_velocities)
            contact_pt_velocity = np.array(contact_pt_velocity)
            contact_pt_loc = np.array(contact_pt_loc)

            if self.make_video:
                writer.cleanup()
                if len(collision_forces) <= 0:
                    os.remove(os.path.join("media", "video_{:d}.mp4".format(i)))

            if self.show_GUI and len(collision_forces) > 0:
                self.plot_collision_forces(
                    collision_forces,
                    robot_target_velocities,
                    robot_cmd_velocities,
                    contact_pt_velocity,
                    contact_pt_loc
                )
            return collision_forces

    def __setup_world(self):
        if self.show_GUI:
            self.physics_client_id = p.connect(p.GUI)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setTimeStep(self.timestep, self.physics_client_id)

        # Insert objects
        self.robot = self.Robot(self.physics_client_id, fixedBase=1, timestep=self.timestep)
        self.human = self.Human(self.physics_client_id, partitioned=True, timestep=self.timestep)
        self.ground = Ground(self.physics_client_id, os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

        if self.show_GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.7, -30, -5, [0, 0, 0.8], self.physics_client_id)

        # Attach Collision Detector and Controller
        self.collider = Collision(
            self.physics_client_id,
            robot=self.robot,
            human=self.human,
            timestep=self.timestep,
            ftsensor_loc=self.robot.ftsensor_loc,
        )
        self.controller = self.Controller(
            v_max=self.robot.v_max,
            omega_max=self.robot.omega_max,
            timestep=self.timestep,
            bumper_r=self.robot.bumper_r,
            bumper_l=self.robot.bumper_l,
        )

    def __step(self,
               collision_forces,
               robot_target_velocities,
               robot_cmd_velocities,
               contact_pt_velocity,
               contact_pt_loc):
        self.robot.advance()
        xyz, quaternion = p.invertTransform(self.robot.global_xyz, self.robot.global_quaternion)
        self.human.advance(xyz, quaternion)
        self.ground.advance(xyz, quaternion)

        p.stepSimulation()

        F = self.collider.get_collision_force()

        if F is not None:
            # ---- Collision Detected ----

            # Update timesteps
            self.robot.timestep = self.collision_timestep
            self.human.timestep = self.collision_timestep
            self.controller.timestep = self.collision_timestep
            self.collider.timestep = self.collision_timestep
            p.setTimeStep(self.collision_timestep, self.physics_client_id)

            # Control Step
            (v, omega) = self.controller.update(
                F=F,
                v_prev=self.robot.v + self.collider.delta_v,
                omega_prev=self.robot.omega + self.collider.delta_omega,
                v_cmd=self.cmd_robot_speed[0],
                omega_cmd=self.cmd_robot_speed[1],
            )
            self.robot.set_speed(v, omega)
            self.human.fix()

            # Store data
            robot_cmd_velocities.append(self.cmd_robot_speed)
            collision_forces.append(np.hstack([F, self.controller._Fmag]))
            robot_target_velocities.append([self.robot.v, self.robot.omega])
            try:
                contact_pt_velocity.append(self.controller.V_contact)
                contact_pt_loc.append(self.controller._theta * 180 / np.pi)
            except Exception:
                pass

            if np.isnan(self.controller.V_contact):
                # Collision has gone below threshold
                # self.robot.set_speed(0, 0)
                self.robot.set_speed(*self.cmd_robot_speed)
                self.collision_over = True

            return self.collision_timestep
        else:
            if len(collision_forces) > 0:
                # No collision after collision has occured
                # self.robot.set_speed(0, 0)
                self.robot.set_speed(*self.cmd_robot_speed)
                self.collision_over = True
            return self.timestep
