import os
import math

import pybullet as p
import numpy as np


def quaternion_multiplication(q1_, q2_):
    q1 = [q1_[3], q1_[0], q1_[1], q1_[2]]
    q2 = [q2_[3], q2_[0], q2_[1], q2_[2]]
    q1_times_q2 = np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])
    return np.roll(q1_times_q2, -1)


def quaternion_and_its_derivative_to_angular_velocity(quaternion, derivative):
    _, inverse_quaternion = p.invertTransform([0, 0, 0], quaternion)
    omega4vec = quaternion_multiplication(2*derivative, inverse_quaternion)
    return np.array([omega4vec[0], omega4vec[1], omega4vec[2]])


def generateQuaternionFromMMMRxRyRz(rx, ry, rz):
    q_intermediate = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
    _, q_intermediate_inv = p.invertTransform([0, 0, 0], q_intermediate)
    q_tmp = p.getQuaternionFromEuler([ry, rz, rx])
    _, q_tmp = p.multiplyTransforms([0, 0, 0], q_intermediate, [0, 0, 0], q_tmp)
    _, q = p.multiplyTransforms([0, 0, 0], q_tmp, [0, 0, 0], q_intermediate_inv)
    return q


def applyMMMRotationToURDFJoint(urdf_body_id, joint_index, rx, ry, rz, inverse=False):
    q = generateQuaternionFromMMMRxRyRz(rx, ry, rz)
    quat_tf_urdf = p.getQuaternionFromEuler([-math.pi/2, math.pi, 0])
    translation, quat_tf_urdf_inv = p.invertTransform([0, 0, 0], quat_tf_urdf)
    _, q = p.multiplyTransforms([0, 0, 0], quat_tf_urdf, [0, 0, 0], q)
    _, q = p.multiplyTransforms([0, 0, 0], q, [0, 0, 0], quat_tf_urdf_inv)

    if inverse:
        _, q = p.invertTransform([0, 0, 0], q)

    p.resetJointStateMultiDof(urdf_body_id, joint_index, q)


class Human:
    """Base Class for Human"""
    gait_phase_step = 0

    def __init__(
        self,
        pybtPhysicsClient,
        folder,
        timestep=0.01,
        scaling=1.0,
        translation_scaling=0.95,   # this is a calibration/scaling of the mocap velocities
    ):
        self.scaling = scaling
        self.setColor()

        # Timestep
        self.timestep = timestep

        # pose containers
        self.is_fixed = False
        self.global_xyz = np.zeros(3)
        self.global_quaternion = p.getQuaternionFromEuler(np.zeros(3))
        self.other_xyz = np.zeros(3)
        self.other_rpy = np.zeros(3)
        self.joint_positions = np.zeros(44)

        # gait motion data
        self.cyclic_joint_positions = np.load(os.path.join(
            folder,
            "walk",
            "cyclic_joint_positions.npy"
        ))
        self.cyclic_pelvis_rotations = np.load(os.path.join(
            folder,
            "walk",
            "cyclic_pelvis_rotations.npy"
        ))
        self.cyclic_pelvis_forward_velocity = scaling * translation_scaling * np.load(os.path.join(
            folder,
            "walk",
            "cyclic_pelvis_forward_velocity.npy"
        ))
        self.cyclic_pelvis_lateral_position = scaling * translation_scaling * np.load(os.path.join(
            folder,
            "walk",
            "cyclic_pelvis_lateral_position.npy"
        ))
        self.cyclic_pelvis_vertical_position = scaling * translation_scaling * np.load(os.path.join(
            folder,
            "walk",
            "cyclic_pelvis_vertical_position.npy"
        ))
        self.cycle_time_steps = np.load(os.path.join(
            folder,
            "walk",
            "cycle_time_steps.npy"
        ))

        self.resetGlobalTransformation()

    def setColor(self):
        # arm_indices, foot_indices ... = ...
        cl = [
            [116,  66, 200],  # purple heart            # noqa: E201,E241
            [252, 116, 253],  # pink flamingo           # noqa: E201,E241
            [242,  40,  71],  # scarlet                 # noqa: E201,E241
            [255, 127,   0],  # orange                  # noqa: E201,E241
            [253, 252, 116],  # unmellow yellow         # noqa: E201,E241
            [190, 192,  10],  # mellow green (unused)   # noqa: E201,E241
            [ 29, 249,  20],  # electric lime           # noqa: E201,E241
            [120, 219, 226],  # aquamarine              # noqa: E201,E241
            [ 59, 176, 143],  # jungle green            # noqa: E201,E241
            [221, 148, 117],  # copper                  # noqa: E201,E241
            [  0,   0,   0],  # black                   # noqa: E201,E241
            [230, 230, 230],  # grey white              # noqa: E201,E241
            [  0,   0, 255],  # blue                    # noqa: E201,E241
        ]
        link_color_index_map = [
            0 , 0 , 0, 	# chest belly pelvis (front)    # noqa: E203
            12, 12, 	# upper legs                    # noqa: E203
            7 , 7 ,     # shins                         # noqa: E203
            6 , 6 , 	# ankles/feet                   # noqa: E203
            1 , 1 , 	# upper arms                    # noqa: E203
            1 , 1 , 	# forearms                      # noqa: E203
            1 , 1 , 	# hands                         # noqa: E203
            10, 		# neck (front)                  # noqa: E203
            4 , 		# head (front/face)             # noqa: E203
            6 , 6 , 	# soles/feet                    # noqa: E203
            6 , 6 , 	# toes/feet                     # noqa: E203
            0 , 0 , 0,  # chest belly pelvis (back)     # noqa: E203
            9 , 		# neck (back)                   # noqa: E203
            11 			# head (back/skull)             # noqa: E203
        ]

        sdl = p.getVisualShapeData(self.body_id)
        for i in range(len(sdl)):
            j = link_color_index_map[i]
            p.changeVisualShape(
                self.body_id,
                sdl[i][1],
                rgbaColor=[cl[j][0]/255, cl[j][1]/255, cl[j][2]/255, 1]
            )

    def resetGlobalTransformation(self,
                                  xyz=np.zeros(3),
                                  rpy=np.zeros(3),
                                  gait_phase_value=0):
        self.initial_xyz = np.array(xyz)
        self.initial_rpy = np.array(rpy) + np.array([0, 0, -np.pi/2])
        self.other_xyz[0] = 0.0
        self.setGaitPhase(gait_phase_value)

    def setGaitPhase(self, period_fraction):
        period_fraction = abs(period_fraction) - int(abs(period_fraction))
        self.gait_phase_step = int(period_fraction*np.size(self.cycle_time_steps))

        self.other_xyz[1] = self.cyclic_pelvis_lateral_position[self.gait_phase_step]
        self.other_xyz[2] = self.cyclic_pelvis_vertical_position[self.gait_phase_step]

        self.other_rpy[:] = self.cyclic_pelvis_rotations[:, self.gait_phase_step]

        self.joint_positions[:] = self.cyclic_joint_positions[:, self.gait_phase_step]

        self.__apply_pose()

    def advance(self, global_xyz, global_quaternion):
        self.global_xyz = global_xyz
        self.global_quaternion = global_quaternion

        if not self.is_fixed:
            self.other_xyz[:] += self.cyclic_pelvis_forward_velocity[self.gait_phase_step]*self.timestep

            self.gait_phase_step += 1
            if self.gait_phase_step == np.size(self.cycle_time_steps):
                self.gait_phase_step = 0

            self.other_xyz[1] = self.cyclic_pelvis_lateral_position[self.gait_phase_step]
            self.other_xyz[2] = self.cyclic_pelvis_vertical_position[self.gait_phase_step]

            self.other_rpy[:] = self.cyclic_pelvis_rotations[:, self.gait_phase_step]

            self.joint_positions[:] = self.cyclic_joint_positions[:, self.gait_phase_step]

        self.__apply_pose()

    def fix(self):
        self.is_fixed = True

    def reset(self):
        self.is_fixed = False

    def regress(self):
        self.other_xyz[:] -= self.cyclic_pelvis_forward_velocity[self.gait_phase_step]*0.01

        self.gait_phase_step -= 1
        if self.gait_phase_step == -1:
            self.gait_phase_step = np.size(self.cycle_time_steps)-1

        self.other_xyz[1] = self.cyclic_pelvis_lateral_position[self.gait_phase_step]
        self.other_xyz[2] = self.cyclic_pelvis_vertical_position[self.gait_phase_step]

        self.other_rpy[:] = self.cyclic_pelvis_rotations[:, self.gait_phase_step]

        self.joint_positions[:] = self.cyclic_joint_positions[:, self.gait_phase_step]

        self.__apply_pose()

    def set_body_velocities_from_gait(self):
        self.advance()

        pos_2, ori_2 = p.getBasePositionAndOrientation(self.body_id)
        joint_position_list_2 = self.__get_joint_positions_as_list()

        self.regress()
        self.regress()

        pos_1, ori_1 = p.getBasePositionAndOrientation(self.body_id)
        joint_position_list_1 = self.__get_joint_positions_as_list()

        self.advance()

        # compute base velocities via central differences assuming a timestep of 0.01 [s]
        baseLinearVelocity = (np.array(pos_2) - np.array(pos_1))/0.02
        quaternion_derivative = (np.array(ori_2) - np.array(ori_1))/0.02

        _, ori_now = p.getBasePositionAndOrientation(self.body_id)
        _, inverse_ori_now = p.invertTransform([0, 0, 0], ori_now)
        omega4vec = quaternion_multiplication(2*quaternion_derivative, inverse_ori_now)
        baseAngularVelocity = [omega4vec[0], omega4vec[1], omega4vec[2]]

        p.resetBaseVelocity(self.body_id,
                            linearVelocity=baseLinearVelocity,
                            angularVelocity=baseAngularVelocity)

        # compute joint velocities via central differences assuming a timestep of 0.01 [s]
        joint_position_list_now = self.__get_joint_positions_as_list()
        for i in range(len(joint_position_list_now)):
            joint_info = p.getJointInfo(self.body_id, i)
            if joint_info[2] == p.JOINT_SPHERICAL:
                omega = quaternion_and_its_derivative_to_angular_velocity(
                    joint_position_list_now[i],
                    (np.array(joint_position_list_2[i]) - np.array(joint_position_list_1[i]))/0.02
                )
                p.resetJointStateMultiDof(
                    self.body_id,
                    i,
                    targetValue=joint_position_list_now[i],
                    targetVelocity=omega
                )
            elif joint_info[2] == p.JOINT_REVOLUTE:
                omega_scalar = (joint_position_list_2[i] - joint_position_list_1[i])/0.02
                p.resetJointState(
                    self.body_id,
                    i,
                    targetValue=joint_position_list_now[i],
                    targetVelocity=omega_scalar
                )

    def __get_joint_positions_as_list(self):
        joint_position_list = []
        for i in range(p.getNumJoints(self.body_id)):
            joint_info = p.getJointInfo(self.body_id, i)
            if joint_info[2] == p.JOINT_SPHERICAL:
                joint_state = p.getJointStateMultiDof(self.body_id, i)
            elif joint_info[2] == p.JOINT_REVOLUTE:
                joint_state = p.getJointState(self.body_id, i)
            joint_position_list.append(joint_state[0])
        return joint_position_list

    def __apply_pose(self):
        # chest to belly
        applyMMMRotationToURDFJoint(self.body_id, 0,
                                    self.joint_positions[6],
                                    self.joint_positions[7],
                                    self.joint_positions[8],
                                    inverse=True)

        # belly to pelvis
        applyMMMRotationToURDFJoint(self.body_id, 1,
                                    self.joint_positions[3],
                                    self.joint_positions[4],
                                    self.joint_positions[5],
                                    inverse=True)

        # pelvis to right leg
        applyMMMRotationToURDFJoint(self.body_id, 2,
                                    self.joint_positions[33],
                                    self.joint_positions[34],
                                    self.joint_positions[35])

        # pelvis to left leg
        applyMMMRotationToURDFJoint(self.body_id, 3,
                                    self.joint_positions[17],
                                    self.joint_positions[18],
                                    self.joint_positions[19])

        # right leg to right shin
        p.resetJointState(self.body_id, 4, -self.joint_positions[36])

        # left leg to left shin
        p.resetJointState(self.body_id, 5, -self.joint_positions[20])

        # right shin to right foot
        applyMMMRotationToURDFJoint(self.body_id, 6,
                                    self.joint_positions[28],
                                    self.joint_positions[29],
                                    self.joint_positions[30])

        # left shin to left foot
        applyMMMRotationToURDFJoint(self.body_id, 7,
                                    self.joint_positions[12],
                                    self.joint_positions[13],
                                    self.joint_positions[14])

        # chest_to_right_arm
        applyMMMRotationToURDFJoint(self.body_id, 8,
                                    self.joint_positions[37],
                                    self.joint_positions[38],
                                    self.joint_positions[39])

        # chest_to_left_arm
        applyMMMRotationToURDFJoint(self.body_id, 9,
                                    self.joint_positions[21],
                                    self.joint_positions[22],
                                    self.joint_positions[23])

        # right arm to right forearm
        p.resetJointState(self.body_id, 10, -self.joint_positions[31])

        # left arm to left forearm
        p.resetJointState(self.body_id, 11, -self.joint_positions[15])

        # right_forearm_to_right_hand
        applyMMMRotationToURDFJoint(self.body_id, 12,
                                    self.joint_positions[40],
                                    self.joint_positions[41],
                                    0.0)

        # left_forearm_to_left_hand
        applyMMMRotationToURDFJoint(self.body_id, 13,
                                    self.joint_positions[24],
                                    self.joint_positions[25],
                                    0.0)

        # chest_to_neck
        applyMMMRotationToURDFJoint(self.body_id, 14,
                                    self.joint_positions[0],
                                    self.joint_positions[1],
                                    self.joint_positions[2])

        # neck_to_head
        applyMMMRotationToURDFJoint(self.body_id, 15,
                                    self.joint_positions[9],
                                    self.joint_positions[10],
                                    self.joint_positions[11])

        # right foot to right sole
        p.resetJointState(self.body_id, 16, self.joint_positions[43])

        # left foot to left sole
        p.resetJointState(self.body_id, 17, self.joint_positions[27])

        # right sole to right toes
        p.resetJointState(self.body_id, 18, -self.joint_positions[42])

        # left sole to left toes
        p.resetJointState(self.body_id, 19, -self.joint_positions[26])

        # Base rotation and Zero Translation (for now)
        self.__applyMMMRotationAndZeroTranslationToURDFBody(
            self.other_rpy[0],
            self.other_rpy[1],
            self.other_rpy[2]
        )

        # Base translation
        self.__applyMMMTranslationToURDFBody(
            self.other_xyz[0],
            self.other_xyz[1],
            self.other_xyz[2]
        )

    def __applyMMMRotationAndZeroTranslationToURDFBody(self, rx, ry, rz):
        # call this function AFTER applying the BT- and BP-joint angles for the urdf (as shown above)

        # get the rotation from the (hypothetical) world to the pelvis
        tpcom, rpcom, tplcom, rplcom, tpf, rotation_pelvis_frame, v, omega = p.getLinkState(
            self.body_id, 1, True, True
        )

        # get the rotation from the (hypothetical) world to the chest
        t_com, r_com = p.getBasePositionAndOrientation(self.body_id)
        # m, lfr, lI, lIt, lIr, rest, rfr, sfr, cd, cs = p.getDynamicsInfo(bodyUniqueId,-1)
        # t_tmp, r_tmp = p.invertTransform([0,0,0], lIr)
        # t_tmp, rotation_chest_frame = p.multiplyTransforms([0,0,0], r_com, [0,0,0], r_tmp)
        rotation_chest_frame = r_com

        # get the rotation from the pelvis to the chest
        t_tmp, r_tmp = p.invertTransform([0, 0, 0], rotation_pelvis_frame)
        t_tmp, r_pelvis_to_chest = p.multiplyTransforms(
            [0, 0, 0], r_tmp,
            [0, 0, 0], rotation_chest_frame
        )

        # pre-multiply with rotation from MMM to my axis convention
        t_tmp, r_mmm_pelvis_to_chest = p.multiplyTransforms(
            [0, 0, 0], p.getQuaternionFromEuler([-math.pi/2, math.pi, 0]),
            [0, 0, 0], r_pelvis_to_chest
        )

        # generate rotation from world to pelvis MMM frame
        # r_world_to_mmm_pelvis = generateQuaternionFromMMMRxRyRz(rx, ry, rz)
        # Alternative (Z-Y-X order of rotations, not like for joints):
        r_world_to_mmm_pelvis = p.getQuaternionFromEuler([rx, ry, rz])
        # r_world_to_mmm_pelvis = generateQuaternionFromMMMGlobalRxRyRz(rx, ry, rz)

        # compose rotation from world to my chest frame
        t_tmp, r_world_to_chest = p.multiplyTransforms(
            [0, 0, 0], r_world_to_mmm_pelvis,
            [0, 0, 0], r_mmm_pelvis_to_chest
        )

        # pre-multiply with global transform
        # t_final, r_final = p.multiplyTransforms([self.global_xyz[0], self.global_xyz[1], self.global_xyz[2]],
        # p.getQuaternionFromEuler([self.global_rpy[0], self.global_rpy[1], self.global_rpy[2]]),
        # [0, 0, 0], r_world_to_chest)

        # apply it to the base together with a zero translation
        p.resetBasePositionAndOrientation(self.body_id, [100, 100, 100], r_world_to_chest)

    def __applyMMMTranslationToURDFBody(self, tx, ty, tz):

        # get the translation to the left leg frame
        tllcom, rllcom, tlllcom, rlllcom, translation_to_left_leg_frame, rllf, vll, omegall = p.getLinkState(
            self.body_id, 3, True, True
        )

        # get the translation to the right leg frame
        trlcom, rrlcom, trllcom, rrllcom, translation_to_right_leg_frame, rrlf, vrl, omegarl = p.getLinkState(
            self.body_id, 2, True, True
        )

        t_base_com, r_base_com = p.getBasePositionAndOrientation(self.body_id)

        t_phb_center_to_base_com = [0, 0, 0]
        for i in range(3):
            t_phb_center_to_base_com[i] = t_base_com[i] - 0.5*(
                translation_to_right_leg_frame[i] + translation_to_left_leg_frame[i])

        t_base_com = [
            tx + t_phb_center_to_base_com[0],
            ty + t_phb_center_to_base_com[1],
            tz + t_phb_center_to_base_com[2]]

        # pre-multiply with global transform
        t_global, r_global = p.multiplyTransforms(
            self.global_xyz,
            self.global_quaternion,
            self.initial_xyz,
            p.getQuaternionFromEuler(self.initial_rpy),
        )
        t_final, r_final = p.multiplyTransforms(
            t_global,
            r_global,
            t_base_com,
            r_base_com,
        )

        # apply it to the base together with a zero translation
        p.resetBasePositionAndOrientation(self.body_id, t_final, r_final)
