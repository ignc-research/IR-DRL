from email.mime import base
import os
import pybullet as p
import pybullet_data as pd
import numpy as np
import time


def getinversePoisition(robot_id, base_link, effector_link, position_desired,orientation_desired=[]):
    joints_info = []
    joint_damping = []
    joint_ll = []
    joint_ul = []
    useOrientation=len(orientation_desired)
    for i in range(effector_link):
        joints_info.append(p.getJointInfo(robot_id, i))
    robotEndEffectorIndex = effector_link
    numJoints = p.getNumJoints(robot_id)
    useNullSpace = 1
    ikSolver = 1
    pos = [position_desired[0], position_desired[1], position_desired[2]]
    # end effector points down, not up (in case useOrientation==1)
    if useOrientation:
        orn = p.getQuaternionFromEuler([orientation_desired[0],orientation_desired[1] , orientation_desired[2]])
    if (useNullSpace == 1):
        if (useOrientation > 0):
            jointPoses = p.calculateInverseKinematics(robot_id, robotEndEffectorIndex, pos, orn)
        else:
            jointPoses = p.calculateInverseKinematics(robot_id,
                                                robotEndEffectorIndex,
                                                pos,
                                                lowerLimits=joint_ll,
                                                upperLimits=joint_ul,
                                            )
            
        # print(jointPoses)
    else:
        if (useOrientation > 0):
            jointPoses = p.calculateInverseKinematics(robot_id,
                                                    robotEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    solver=ikSolver,
                                                    maxNumIterations=100,
                                                    residualThreshold=.01)
            # print(jointPoses)
        else:
            jointPoses = p.calculateInverseKinematics(robot_id,
                                                    robotEndEffectorIndex,
                                                    pos,
                                                    solver=ikSolver)
            p.getJointInfo()
    return jointPoses

def go_to_target(robotid, base_link, effector_link, position, orientation):
    jointPoses = getinversePoisition(robotid, base_link, effector_link, position, orientation)
    for i in range(base_link, effector_link):
        # p.setJointMotorControl2(bodyIndex=robotid,
        #                         jointIndex=i,
        #                         controlMode=p.POSITION_CONTROL,
        #                         targetPosition=jointPoses[i-1],
        #                         targetVelocity=0,
        #                         force=500,
        #                         positionGain=0.03,
        #                         velocityGain=1)
    # print (p.getLinkState(RobotUid,7))
        p.resetJointState(bodyUniqueId=robotid,
                                jointIndex=i,
                                targetValue=jointPoses[i-1],
                                )
    return

def go_to_target_kuka(robotid, base_link, effector_link, position, orientation):
    jointPoses = getinversePoisition(robotid, base_link, effector_link, position, orientation)
    for i in range(base_link, effector_link):
        p.resetJointState(bodyUniqueId=robotid,
                                jointIndex=i,
                                targetValue=jointPoses[i],
                                )
    return