import os
import pybullet as p
import pybullet_data as pd
import numpy as np

class MotionExecute():
    def __init__(
        self,
        robot_id, 
        base_link,
        effector_link
        ):
        self.robot_id = robot_id
        self.base_link = base_link
        self.effector_link = effector_link
        
    def getinversePoisition(self, position_desired,orientation_desired=[]):
        joints_info = []
        joint_ll = []
        joint_ul = []
        useOrientation=len(orientation_desired)
        for i in range(self.effector_link+1):
            joints_info.append(p.getJointInfo(self.robot_id, i))
        robotEndEffectorIndex = self.effector_link
        numJoints = p.getNumJoints(self.robot_id)
        useNullSpace = 1
        ikSolver = 1
        pos = [position_desired[0], position_desired[1], position_desired[2]]
        # end effector points down, not up (in case useOrientation==1)
        if useOrientation:
            orn = p.getQuaternionFromEuler([orientation_desired[0],orientation_desired[1] , orientation_desired[2]])
        if (useNullSpace == 1):
            if (useOrientation > 0):
                jointPoses = p.calculateInverseKinematics(self.robot_id, robotEndEffectorIndex, pos, orn)
            else:
                jointPoses = p.calculateInverseKinematics(self.robot_id,
                                                    robotEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=joint_ll,
                                                    upperLimits=joint_ul,
                                                )
                
            # print(jointPoses)
        else:
            if (useOrientation > 0):
                jointPoses = p.calculateInverseKinematics(self.robot_id,
                                                        robotEndEffectorIndex,
                                                        pos,
                                                        orn,
                                                        solver=ikSolver,
                                                        maxNumIterations=100,
                                                        residualThreshold=.01)
                # print(jointPoses)
            else:
                jointPoses = p.calculateInverseKinematics(self.robot_id,
                                                        robotEndEffectorIndex,
                                                        pos,
                                                        solver=ikSolver)
                p.getJointInfo()
        return jointPoses

    def go_to_target(self, position, orientation):
        jointPoses = self.getinversePoisition(position, orientation)
        for i in range(self.base_link, self.effector_link):
            p.resetJointState(bodyUniqueId=self.robot_id,
                                    jointIndex=i,
                                    targetValue=jointPoses[i-1],
                                    )
        return