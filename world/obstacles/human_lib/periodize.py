#PYHTON 3!

import xml.etree.ElementTree as ET
import numpy as np
import re
import matplotlib.pyplot as plt
import time

#the following file is from https://motion-database.humanoids.kit.edu/details/motions/37/
tree = ET.parse('../data/WalkingStraightForwards05.xml')
root = tree.getroot()

time_trajectory = np.zeros([len(root[0][3]), 1])
joint_position_trajectories = np.zeros([len(root[0][3]), 44])
pose_trajectories = np.zeros([len(root[0][3]), 6])

initial_position_value_string_list = re.split(' ', root[0][3][0][1].text)
x0 = float(initial_position_value_string_list[0])/1000
y0 = float(initial_position_value_string_list[1])/1000
z0 = float(initial_position_value_string_list[2])/1000

for t in range(len(root[0][3])):
	value_string_list = re.split(' ', root[0][3][t][3].text)

	# store the values for plotting
	for j in range(44):
		joint_position_trajectories[t, j] = float(value_string_list[j])

	position_value_string_list = re.split(' ', root[0][3][t][1].text)
	rotation_value_string_list = re.split(' ', root[0][3][t][2].text)

	pose_trajectories[t, 0] = float(position_value_string_list[0])
	pose_trajectories[t, 1] = float(position_value_string_list[1])
	pose_trajectories[t, 2] = float(position_value_string_list[2])
	pose_trajectories[t, 3] = float(rotation_value_string_list[0])
	pose_trajectories[t, 4] = float(rotation_value_string_list[1])
	pose_trajectories[t, 5] = float(rotation_value_string_list[2])

	time_trajectory[t] = float(root[0][3][t][0].text)

show_original_signals = True
if show_original_signals:	
	for j in range(44):
		plt.plot(joint_position_trajectories[:, j])
	plt.title("Joint positions")
	plt.show()

	plt.plot(pose_trajectories[:, 0], 'r')
	plt.plot(pose_trajectories[:, 1], 'g')
	plt.plot(pose_trajectories[:, 2], 'b')
	plt.title("Position XYZ")
	plt.show()

	plt.plot(pose_trajectories[:, 3], 'r')
	plt.plot(pose_trajectories[:, 4], 'g')
	plt.plot(pose_trajectories[:, 5], 'b')
	plt.title("Rotation XYZ")
	plt.show()

shape = np.shape(pose_trajectories[:, 0])
velocities = np.zeros([shape[0],3])
for i in range(1,shape[0]-1):
	for j in range(3):
		velocities[i, j] = (pose_trajectories[i+1, j] - pose_trajectories[i-1, j])/(
			time_trajectory[i+1] - time_trajectory[i-1])

show_velocities = True
if show_velocities:
	plt.plot(velocities[:, 0], 'r')
	plt.plot(velocities[:, 1], 'g')
	plt.plot(velocities[:, 2], 'b')
	plt.title("Velocity XYZ")
	plt.show()

# Crop window for WalkingStraightForwards05
T = range(231, 356) #~1 period
J = joint_position_trajectories[T,:]
for j in range(44):
	plt.plot(J[:, j])
plt.title("Joint positions in time window")
plt.show()

# compute the Joint positions' FFTs
JFFTs = np.zeros(np.shape(J), dtype=np.cdouble)
for j in range(44):
	JFFTs[:,j] = np.fft.fft(J[:,j])
	plt.plot(np.power(np.absolute(JFFTs[1:int(len(T)/2),j]), 2))
plt.title("FFT of joint positions in time window")
plt.show()

# remove frequencies and transform back
JProcessed = np.zeros(np.shape(J))
for j in range(44):
	for i in range(5, len(T)-4):
		JFFTs[i,j] = 0.0
	JProcessed[:,j] = np.fft.ifft(JFFTs[:,j])
	plt.plot(JProcessed[:, j],':')
plt.title("Processed joint positions in time window")
plt.show()

#timestep = 0.01
#freq = np.fft.fftfreq(len(T), d=timestep)
#print(np.power(freq,-1))

# compute the base velocities' FFTs
VB = velocities[T,:]
VBFFT = np.zeros(np.shape(VB), dtype=np.cdouble)
for j in range(3):
	VBFFT[:,j] = np.fft.fft(VB[:,j])
plt.plot(np.power(np.absolute(VBFFT[1:int(len(T)/2),0]), 2), 'r')
plt.plot(np.power(np.absolute(VBFFT[1:int(len(T)/2),1]), 2), 'g')
plt.plot(np.power(np.absolute(VBFFT[1:int(len(T)/2),2]), 2), 'b')
plt.title("FFT of base velocities in time window")
plt.show()

# remove frequencies and transform back
VBProcessed = np.zeros(np.shape(VB))
for j in range(3):
	for i in range(5, len(T)-4):
		VBFFT[i,j] = 0.0
	VBProcessed[:,j] = np.fft.ifft(VBFFT[:,j])
plt.plot(VBProcessed[:, 0],'r:')
plt.plot(VBProcessed[:, 1],'g:')
plt.plot(VBProcessed[:, 2],'b:')
plt.title("Processed base velocities in time window")
plt.show()

# compute the base rotations' FFTs
RB = pose_trajectories[T,3:]
RBFFT = np.zeros(np.shape(RB), dtype=np.cdouble)
for j in range(3):
	RBFFT[:,j] = np.fft.fft(RB[:,j])
plt.plot(np.power(np.absolute(RBFFT[1:int(len(T)/2),0]), 2), 'r')
plt.plot(np.power(np.absolute(RBFFT[1:int(len(T)/2),1]), 2), 'g')
plt.plot(np.power(np.absolute(RBFFT[1:int(len(T)/2),2]), 2), 'b')
plt.title("FFT of base rotations in time window")
plt.show()

# remove frequencies and transform back
RBProcessed = np.zeros(np.shape(RB))
for j in range(3):
	for i in range(5, len(T)-4):
		RBFFT[i,j] = 0.0
	RBProcessed[:,j] = np.fft.ifft(RBFFT[:,j])
plt.plot(RBProcessed[:, 0],'r:')
plt.plot(RBProcessed[:, 1],'g:')
plt.plot(RBProcessed[:, 2],'b:')
plt.title("Processed base rotations in time window")
plt.show()

# extract the periodic lateral and vertical motion
# compute the base positions' FFTs
PB = pose_trajectories[T,0:3]
PBFFT = np.zeros(np.shape(PB), dtype=np.cdouble)
for j in range(3):
	PBFFT[:,j] = np.fft.fft(PB[:,j])
plt.plot(np.power(np.absolute(PBFFT[1:int(len(T)/2),0]), 2), 'r')
plt.plot(np.power(np.absolute(PBFFT[1:int(len(T)/2),1]), 2), 'g')
plt.plot(np.power(np.absolute(PBFFT[1:int(len(T)/2),2]), 2), 'b')
plt.title("FFT of base positions in time window")
plt.show()

# remove frequencies and transform back
PBProcessed = np.zeros(np.shape(PB))
for j in range(3):
	for i in range(5, len(T)-4):
		PBFFT[i,j] = 0.0
	PBProcessed[:,j] = np.fft.ifft(PBFFT[:,j])
plt.plot(PBProcessed[:, 0],'r:')
plt.plot(PBProcessed[:, 1],'g:')
plt.plot(PBProcessed[:, 2],'b:')
plt.title("Processed base postions in time window")
plt.show()

# Animate joints
import pybullet as p
import math

def generateQuaternionFromMMMRxRyRz(rx, ry, rz):
	q_intermediate = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
	t, q_intermediate_inv = p.invertTransform([0,0,0], q_intermediate)
	q_tmp = p.getQuaternionFromEuler([ry, rz, rx])
	t, q_tmp = p.multiplyTransforms([0,0,0], q_intermediate, [0,0,0], q_tmp)
	t, q = p.multiplyTransforms([0,0,0], q_tmp, [0,0,0], q_intermediate_inv)
	return q

def applyMMMRotationToURDFJoint(urdf_body_id, joint_index, rx, ry, rz, inverse=False):
	q = generateQuaternionFromMMMRxRyRz(rx, ry, rz)
	quat_tf_urdf = p.getQuaternionFromEuler([-math.pi/2, math.pi, 0])
	translation, quat_tf_urdf_inv = p.invertTransform([0,0,0], quat_tf_urdf)
	t, q = p.multiplyTransforms([0,0,0], quat_tf_urdf, [0,0,0], q)
	t, q = p.multiplyTransforms([0,0,0], q, [0,0,0], quat_tf_urdf_inv)

	if inverse:
		t, q = p.invertTransform([0,0,0], q)

	p.resetJointStateMultiDof(urdf_body_id, joint_index, q)

# call this function AFTER applying the BT- and BP-joint angles for the urdf (using the function above) 
def applyMMMRotationAndZeroTranslationToURDFBody(urdf_body_id, rx, ry, rz):
	# get the rotation from the (hypothetical) world to the pelvis
	tpcom, rpcom, tplcom, rplcom, tpf, rotation_pelvis_frame, v, omega = p.getLinkState(
		urdf_body_id, 1, True, True)

	# get the rotation from the (hypothetical) world to the chest
	t_com, r_com = p.getBasePositionAndOrientation(urdf_body_id)
	#m, lfr, lI, lIt, lIr, rest, rfr, sfr, cd, cs = p.getDynamicsInfo(bodyUniqueId,-1)
	#t_tmp, r_tmp = p.invertTransform([0,0,0], lIr)
	#t_tmp, rotation_chest_frame = p.multiplyTransforms([0,0,0], r_com, 
	#	[0,0,0], r_tmp)
	rotation_chest_frame = r_com

	# get the rotation from the pelvis to the chest
	t_tmp, r_tmp = p.invertTransform([0,0,0], rotation_pelvis_frame)
	t_tmp, r_pelvis_to_chest = p.multiplyTransforms([0,0,0], r_tmp, 
		[0,0,0], rotation_chest_frame)

	# pre-multiply with rotation from MMM to my axis convention
	t_tmp, r_mmm_pelvis_to_chest = p.multiplyTransforms([0,0,0],
		p.getQuaternionFromEuler([-math.pi/2, math.pi, 0]), 
		[0,0,0], r_pelvis_to_chest)

	# generate rotation from world to pelvis MMM frame
	#r_world_to_mmm_pelvis = generateQuaternionFromMMMRxRyRz(rx, ry, rz)
	# Alternative (Z-Y-X order of rotations, not like for joints):
	r_world_to_mmm_pelvis = p.getQuaternionFromEuler([rx, ry, rz])
	#r_world_to_mmm_pelvis = generateQuaternionFromMMMGlobalRxRyRz(rx, ry, rz)

	# compose rotation from world to my chest frame
	t_tmp, r_world_to_chest = p.multiplyTransforms([0,0,0], r_world_to_mmm_pelvis,
		[0,0,0], r_mmm_pelvis_to_chest)

	# apply it to the base together with a zero translation
	p.resetBasePositionAndOrientation(urdf_body_id,
		[0, 0, 0], r_world_to_chest)

def applyMMMTranslationToURDFBody(urdf_body_id, tx, ty, tz):
	# get the translation to the pelvis frame
	#tpcom, rpcom, tplcom, rplcom, translation_to_pelvis_frame, rpf, v, omega = p.getLinkState(
	#	urdf_body_id, 1, True, True)

	# get the translation to the left leg frame
	tllcom, rllcom, tlllcom, rlllcom, translation_to_left_leg_frame, rllf, vll, omegall = p.getLinkState(
		urdf_body_id, 3, True, True)

	# get the translation to the right leg frame
	trlcom, rrlcom, trllcom, rrllcom, translation_to_right_leg_frame, rrlf, vrl, omegarl = p.getLinkState(
		urdf_body_id, 2, True, True)

	t_base_com, r_base_com = p.getBasePositionAndOrientation(urdf_body_id)

	t_phb_center_to_base_com = [0,0,0]
	for i in range(3):
		t_phb_center_to_base_com[i] = t_base_com[i] - 0.5*(
			translation_to_right_leg_frame[i] + translation_to_left_leg_frame[i])

	t_base_com = [
		tx + t_phb_center_to_base_com[0],
		ty + t_phb_center_to_base_com[1],
		tz + t_phb_center_to_base_com[2]]

	p.resetBasePositionAndOrientation(urdf_body_id, t_base_com, r_base_com)

# load the human-adult model
physicsClient = p.connect(p.GUI)
human_adult_ID = p.loadURDF("../data/human_adult_scan.urdf",
	flags=p.URDF_MAINTAIN_LINK_ORDER)

v_average = np.mean(VBProcessed, 0)
v_average_yz = np.array([0.0,v_average[1], v_average[2]])
print (v_average[0],v_average[1],v_average[2])
R_average = np.mean(RBProcessed,0)
print (R_average[0],R_average[1],R_average[2])

create_npy_files = False
if create_npy_files:
	RBProcessedAdjusted = np.copy(RBProcessed)
	RBProcessedAdjusted[:,2] += -R_average[2]+np.pi/2
	np.save("cyclic_joint_positions.npy", np.transpose(JProcessed))
	np.save("cyclic_pelvis_rotations.npy", np.transpose(RBProcessedAdjusted))
	np.save("cyclic_pelvis_forward_velocity.npy", VBProcessed[:,0]/1000.0)
	np.save("cyclic_pelvis_lateral_position.npy", (PBProcessed[:, 1]- PBProcessed[0, 1])/1000.0)
	np.save("cyclic_pelvis_vertical_position.npy", (PBProcessed[:, 2]- PBProcessed[0, 2])/1000.0)
	np.save("cycle_time_steps.npy", time_trajectory[T] - time_trajectory[0])

t = 0
base_position = np.zeros([3,1])
camera_position = np.zeros([3,1])
while True:
	# chest to belly
	applyMMMRotationToURDFJoint(human_adult_ID, 0,
		JProcessed[t,6],
		JProcessed[t,7],
		JProcessed[t,8], inverse = True)

	# belly to pelvis
	applyMMMRotationToURDFJoint(human_adult_ID, 1,
		JProcessed[t,3],
		JProcessed[t,4],
		JProcessed[t,5], inverse = True)

	# pelvis to right leg
	applyMMMRotationToURDFJoint(human_adult_ID, 2,
		JProcessed[t,33],
		JProcessed[t,34],
		JProcessed[t,35])
	# pelvis to left leg
	applyMMMRotationToURDFJoint(human_adult_ID, 3,
		JProcessed[t,17],
		JProcessed[t,18],
		JProcessed[t,19])
	
	# right leg to right shin
	p.resetJointState(human_adult_ID, 4, -JProcessed[t,36])
	# left leg to left shin
	p.resetJointState(human_adult_ID, 5, -JProcessed[t,20])

	# right shin to right foot
	applyMMMRotationToURDFJoint(human_adult_ID, 6,
		JProcessed[t,28],
		JProcessed[t,29],
		JProcessed[t,30])
	# left shin to left foot
	applyMMMRotationToURDFJoint(human_adult_ID, 7,
		JProcessed[t,12],
		JProcessed[t,13],
		JProcessed[t,14])

	# chest_to_right_arm
	applyMMMRotationToURDFJoint(human_adult_ID, 8,
		JProcessed[t,37],
		JProcessed[t,38],
		JProcessed[t,39])

	# chest_to_left_arm
	applyMMMRotationToURDFJoint(human_adult_ID, 9,
		JProcessed[t,21],
		JProcessed[t,22],
		JProcessed[t,23])

	# right arm to right forearm
	p.resetJointState(human_adult_ID, 10, -JProcessed[t,31])
	# left arm to left forearm
	p.resetJointState(human_adult_ID, 11, -JProcessed[t,15])

	# right_forearm_to_right_hand
	applyMMMRotationToURDFJoint(human_adult_ID, 12,
		JProcessed[t,40],
		JProcessed[t,41],
		0.0)

	# left_forearm_to_left_hand
	applyMMMRotationToURDFJoint(human_adult_ID, 13,
		JProcessed[t,24],
		JProcessed[t,25],
		0.0)

	# chest_to_neck
	applyMMMRotationToURDFJoint(human_adult_ID, 14,
		JProcessed[t,0],
		JProcessed[t,1],
		JProcessed[t,2])

	# neck_to_head
	applyMMMRotationToURDFJoint(human_adult_ID, 15,
		JProcessed[t,9],
		JProcessed[t,10],
		JProcessed[t,11])

	# right foot to right sole
	p.resetJointState(human_adult_ID, 16, JProcessed[t,43])
	# left foot to left sole
	p.resetJointState(human_adult_ID, 17, JProcessed[t,27])

	# right sole to right toes
	p.resetJointState(human_adult_ID, 18, -JProcessed[t,42])
	# left sole to left toes
	p.resetJointState(human_adult_ID, 19, -JProcessed[t,26])

	# Base rotation and Zero Translation (for now)
	applyMMMRotationAndZeroTranslationToURDFBody(human_adult_ID,
		RBProcessed[t,0],
		RBProcessed[t,1],
		RBProcessed[t,2]-R_average[2]+np.pi/2)

	# Base translation
	applyMMMTranslationToURDFBody(human_adult_ID,
		base_position[0],
		base_position[1],
		base_position[2])

	p.resetDebugVisualizerCamera(cameraDistance=3, 
		cameraYaw=-90, cameraPitch=-25, cameraTargetPosition=[camera_position[0],
		camera_position[1],camera_position[2]])

	time.sleep(0.01)

	base_position[:,0] = base_position[:,0] + 0.01*(VBProcessed[t,:] - v_average_yz)/1000.0
	#base_position[1:,0] = (PBProcessed[t, 1:]- PBProcessed[0, 1:])/1000.0
	camera_position[:,0] = camera_position[:,0] + 0.01*(v_average-v_average_yz)/1000.0


	t += 1
	if t == len(T):
		t = 0