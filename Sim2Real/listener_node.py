#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import numpy as np
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import sys

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

class listener_node_one:
    def __init__(self, action_rate, control_rate):
        self.end_effector_link_id = 6
        self.end_effector_xyz = None
        self.end_effector_rpy = None
        self.end_effector_quat = None
        self.actions = []  # list of lists
        self.durations = []  # list of floats
        self.joints = None
        self.ee_transform = {
            "translation": [],
            "rotation": []
        }
        #self.model = bla
        self.trajectory_client = actionlib.SimpleActionClient(
            "scaled_pos_joint_traj_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        timeout = rospy.Duration(5)
        if not self.trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

    # verwerten Daten, wandeln in das Format vom NN, fragen NN, wandeln Output vom NN in vom
    # UR5 Driver verstandene Commands
    # Frequenz: festgelegt von uns
    def cbAction(self, event):
        self.actions = [[0, -1.57, -1.57, 0, 0, 0]]
        self.durations = [10]

    def cbControl(self, event):     
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = JOINT_NAMES
        
        for idx, waypoint in enumerate(self.actions):
            point = JointTrajectoryPoint()
            point.positions = waypoint
            point.time_from_start = rospy.Duration(sum(self.durations[:idx+1]))
            goal.trajectory.points.append(point)
        
        self.trajectory_client.send_goal(goal)       
        

    # holen Daten, Frequenz: vom Sender (UR5)
    def cbGetPos(self, data):
        for entry in data.transforms:
            parent_frame_id = entry.header.frame_id
            frame_id = entry.child_frame_id
            if parent_frame_id == "base" and frame_id == "tool0_controller":
                xyz = np.array([entry.transform.translation.x, entry.transform.translation.y, entry.transform.translation.z])
                quat = np.array([entry.transform.rotation.x, entry.transform.rotation.y, entry.transform.rotation.z, entry.transform.rotation.w])
                self.ee_transform["translation"] = xyz
                self.ee_transform["rotation"] = -quat #anpassung an rviz werte

    def cbGetJoints(self, data):
        self.joints = np.array(data.position, dtype=np.float32)
        #print(self.joints)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = listener_node_one(action_rate=100, control_rate=10)
    rospy.Subscriber("/tf", TFMessage, listener.cbGetPos)
    rospy.Subscriber("/joint_states", JointState, listener.cbGetJoints)
    rospy.Timer(rospy.Duration(secs=1/2), listener.cbAction)
    rospy.Timer(rospy.Duration(secs=1/4), listener.cbControl)
    print("listener node initiated")
    rospy.spin()
