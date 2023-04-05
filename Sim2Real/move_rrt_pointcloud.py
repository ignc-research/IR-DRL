#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
import numpy as np
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2
import sys
from modular_drl_env.gym_env.environment import ModularDRLEnv
from modular_drl_env.util.configparser import parse_config
from modular_drl_env.util.rrt import bi_rrt, smooth_path
from modular_drl_env.world.obstacles.shapes import Box
import ros_numpy
import open3d as o3d

import pybullet as pyb

#
#TODO: env einführen
# config schreiben in der wir den UR5 definieren
# world generator 
# trajectory aus step herausziehen und übergeben an UR5

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# startpos in box = 0.13 0.78 0.01
# zielpos in box = 0.77 0.23 0.36
# goal joint angles = [0.16853678 -0.9832662 0.8828659 -1.7676371 -1.5672787 -0.15731794]
# goal joint angles [ 0.05017271 -0.9653352   0.792964   -1.7775062  -1.4203948  -0.15738994]

class listener_node_one:
    def __init__(self, action_rate, control_rate, num_voxels, point_cloud_static):
        # robot data
        self.end_effector_link_id = 6
        self.end_effector_xyz = None
        self.end_effector_rpy = None
        self.end_effector_quat = None
        self.end_effector_xyz_sim = None
        self.actions = []  # list of lists
        self.durations = []  # list of floats
        self.joints = None
        self.goal = None
        self.q_goal = None
        self.max_rrt_steps = 10000 # maximale steps um mind. eine rrt Lösung zu finden
        self.running_rrt = False  # flag for interference between rrt and symsinc to prevent both from running parallel
        self.point_cloud_static = point_cloud_static
        self.static_done = False
        self.num_voxels = num_voxels
        self.color_voxels = True
        self.control_mode = True   # True=pos, False=joints
        self.startup = True
        
        # cbGetPointcloud
        self.points_raw = None

        # cbPointcloudToPybullet
        self.camera_transform_to_pyb_origin = np.eye(4)
        self.camera_transform_to_pyb_origin[:3, 0] = np.array([-1, 0, 0])        # vektor für Kamera x achse in pybullet
        self.camera_transform_to_pyb_origin[:3, 1] = np.array([0, 0, -1])        # vektor für Kamera y achse in pybullet
        self.camera_transform_to_pyb_origin[:3, 2] = np.array([0, -1, 0])        # vektor für Kamera z achse in pybullet
        self.camera_transform_to_pyb_origin[:3, 3] = np.array([0.25, 1.72, 0.4]) #realwerte für camera position aus listener.py nach kalibrieren 
        self.voxel_size = 0.035 # 0.035
        self.robot_voxel_safe_distance = 0.2

        self.trajectory_client = actionlib.SimpleActionClient(
            "scaled_pos_joint_traj_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        timeout = rospy.Duration(5)
        if not self.trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)
        # custom sim2real config parsen
        _, env_config = parse_config("/home/moga/catkin_ws/src/Universal_Robots_ROS_Driver/ur_robot_driver/scripts/sim2real_rrt_karim.yaml", False)
        env_config["env_id"] = 0
        # mit der config env starten
        self.env = ModularDRLEnv(env_config)
        self.virtual_robot = self.env.robots[0] # env initialisiert robot, wird für spätere Verwendung in der virtual_robot variable gespeichert
        self.env.reset()

        self.voxels = []        # voxel objects in pybullet simulation
        self.pos_nowhere= np.array([0,0,-100])
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
        self.probe_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, scale=[self.voxel_size, self.voxel_size, self.voxel_size], color=[1, 1, 1, 0])
        self.probe_voxel.build()
        for i in range(self.num_voxels):
            #new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, halfExtents=[self.voxel_size/2, self.voxel_size/2, self.voxel_size/2], color=np.concatenate((np.random.uniform(size=(3,)), np.ones(1))))
            new_voxel = Box(position=self.pos_nowhere, rotation=[0,0,0,1], trajectory=[], move_step=0, scale=[self.voxel_size, self.voxel_size, self.voxel_size], color=[1, 0, 0, 1])
            self.voxels.append(new_voxel)
            new_voxel.build()
            self.env.world.objects_ids.append(new_voxel.object_id)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)

        # init ros stuff
        rospy.Subscriber("/tf", TFMessage, self.cbGetPos)
        rospy.Subscriber("/joint_states", JointState, self.cbGetJoints)
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.cbGetPointcloud)
        rospy.Timer(rospy.Duration(secs=1/action_rate), self.cbAction)
        rospy.Timer(rospy.Duration(secs=1/control_rate), self.cbControl)
        rospy.Timer(rospy.Duration(secs=1/100), self.cbSimSync) # sync der Simulation mit dem echten Roboter so schnell wie möglich
        rospy.Timer(rospy.Duration(secs=1/200), self.cbPointcloudToPybullet)

        print("[Listener] initialized node")
        rospy.spin() #Lässt rospy immer weiter laufen


    # verwerten Daten, wandeln in das Format vom NN, fragen NN, wandeln Output vom NN in vom
    # UR5 Driver verstandene Commands
    # Frequenz: festgelegt von uns
    def cbAction(self, event):
        if self.joints is not None:  # wait until self.joints is written to for the first time
            if not self.actions and self.goal is not None: #sofern keine actions vorliegen, environment resetten    
                print("[cbAction] starting RRT")
                self.running_rrt = True
                self.virtual_robot.moveto_joints(self.joints, False) #roboter in der simulation an der stelle bewegen an der sich der echte ur5 auch tatsächlich befindet
                self.virtual_robot.joints_sensor.update(0) #joints werden als eigener "sensor" betrachtet in der Simulation, müssen daher auch geupdatet werden
                 # rufen die inverse kinematics auf um goal zu solven, none = keine quaternion
                try:
                    #pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
                    self.actions = bi_rrt(q_start=self.joints, q_goal=self.q_goal, robot=self.virtual_robot, engine=self.env.engine, obstacles_ids=self.env.world.objects_ids, max_steps=self.max_rrt_steps, epsilon=1e-2, goal_bias=0.1, visible=True)  
                    print("Actions: ", self.actions)
                    #pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                except Exception as e:
                    print("[cbAction] desired goal or start in collision!")
                    self.goal = None
                    self.running_rrt = False
                    print("exception ist das:", e)
                    return
                print("[cbAction] starting path smoothing")
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
                self.actions = smooth_path(self.actions, 1e-2, self.virtual_robot, self.env.engine, self.env.world.objects_ids)[1:]  # entry zero is the start node, so we leave it out # node anzahl verringern
                pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
                self.virtual_robot.moveto_joints(self.joints, False)
                self.running_rrt = False
                # jetzt wird RRT berechnet
                if self.actions:
                    print("[cbAction] RRT & smoothing finished")
                else:
                    print("[cbAction] RRT could not find solution within " + str(self.max_rrt_steps) + " steps!")
    

    def cbControl(self, event):  
        if self.startup:
            pass # TODO   
        if self.goal is None:
            print("[cbControl] current (virtual) position: " + str(self.end_effector_xyz_sim))
            print("[cbControl] current (virtual) joint angles: " + str(self.joints))
            inp = input("[cbControl] Choose between joint goal (j) or position goal (p):")               
            if inp == "j":
                inp = input("[cbControl] Enter a goal by putting six float values (joint angles in rad) with a space between: \n")
                #inp = "0.05017271 -0.9653352 0.792964 -1.7775062 -1.4203948 -0.15738994"
                inp = inp.split(" ")
                try:
                    inp = [float(ele) for ele in inp]
                except ValueError:
                    print("[cbControl] input in wrong format, try again!")
                    return
                self.q_goal = np.array(inp)
                self.virtual_robot.moveto_joints(self.q_goal, False)
                self.virtual_robot.position_rotation_sensor.update(0)
                tmp_pos = self.virtual_robot.position_rotation_sensor.position
                self.goal = tmp_pos
            elif inp == "p":
                inp = input("[cbControl] Enter a goal by putting three float values (xyz) with a space between: \n")
                inp = inp.split(" ")
                try:
                    inp = [float(ele) for ele in inp]
                except ValueError:
                    print("[cbControl] input in wrong format, try again!")
                    return
                # check if inverse kinematics can actually reach the xyz pos
                tmp_goal = np.array(inp)
                q_goal = self.virtual_robot._solve_ik(tmp_goal, None)
                self.virtual_robot.moveto_joints(q_goal, False)
                self.virtual_robot.position_rotation_sensor.update(0)
                tmp_pos = self.virtual_robot.position_rotation_sensor.position
                if np.linalg.norm(tmp_pos - tmp_goal) > 5e-2:
                    print("[cbControl] could not find solution via inverse kinematics that is close enough, try another position")
                else:
                    self.goal = tmp_goal
                    self.q_goal = q_goal
            else: 
                print("[cbControl] invalid control mode!")
                return
                
        elif self.actions and not self.running_rrt:
            print("[cbControl] Starting trajectory transmission")
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = JOINT_NAMES
            self.durations = [7 for _ in self.actions]
            
            for idx, waypoint in enumerate(self.actions):
                point = JointTrajectoryPoint()
                point.positions = waypoint
                point.time_from_start = rospy.Duration(sum(self.durations[:idx+1]))
                goal.trajectory.points.append(point)
            
            #Clear cb action
            self.trajectory_client.send_goal(goal)       
            print("[cbControl] Waiting for trajectory to finish")
            self.trajectory_client.wait_for_result()
            print("[cbControl] Trajectory finished")
            self.actions = []
            self.goal = None

    def cbGetPointcloud(self, data):
        np_data = ros_numpy.numpify(data)
         
        points = np.ones((np_data.shape[0], 4))
        points[:, 0] = np_data['x']
        points[:, 1] = np_data['y']
        points[:, 2] = np_data['z']
        self.points_raw = points

    def cbPointcloudToPybullet(self, event):
        # callback for PointcloudToPybullet
        # static = only one update
        # dynamic = based on update frequency
        if self.points_raw is not None:  # catch first time execution scheduling problems
            if self.point_cloud_static:
                if self.static_done:
                    return
                else:
                    self.static_done = True
                    self.PointcloudToVoxel()
                    self.VoxelsToPybullet()
            else:
                self.PointcloudToVoxel()
                self.VoxelsToPybullet()
        
    def PointcloudToVoxel(self):
        if not self.running_rrt:
            # prefilter data
            points = self.points_raw
            points_norms = np.linalg.norm(points, axis=1)
            points = points[np.logical_and(points_norms <= 2.5, points_norms > 0.4)]  # remove data points that are too far or too close

            # rotate and translate the data into the world coordinate system
            points = np.dot(self.camera_transform_to_pyb_origin, points.T).T.reshape(-1,4)
            points = points[:, :3]  # remove homogeneous component

            # postfilter data
            # remove the table, points below lower threshold
            points = points[points[:, 2] > -0.1]
            # remove points above a certain threshold where no obstacles should be :2 weil z achse
            #points = points[points[:, 2] < 0.5]
            # filter objects near endeffector
            ee_pos, ee_rot = self.env.engine.get_link_state(self.virtual_robot.object_id,self.virtual_robot.end_effector_link_id)
            #ee_pos = np.array(pyb.getLinkState(self.engine._robots[self.virtual_robot.object_id], self.virtual_robot.end_effector_link_id)[4])
            # TODO may need to be adjusted with KINECT Cam
            mask_left = self.delete_points_by_circle_center(points, ee_pos + np.array([0.1, -0.1, 0.2]))  # TODO: maybe find a general way to do this
            mask_above = self.delete_points_by_circle_center(points, ee_pos + np.array([0, 0, 0.15]))
            delete_mask = np.logical_and(mask_left, mask_above)
            # filter by bool_array
            points = points[delete_mask] 
            

            # get the offset for the voxel transformation later on
            offset = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])]) #points.min(axis=0)

            pcd = o3d.geometry.PointCloud()
            # convert it into open 3d format
            pcd.points = o3d.utility.Vector3dVector(points)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
            voxel_centers = [voxel.grid_index for voxel in voxel_grid.get_voxels()]
            voxel_centers = np.array(voxel_centers)
            voxel_centers = voxel_centers * self.voxel_size + offset

            delete_mask = np.zeros(shape=(voxel_centers.shape[0],), dtype=bool)
            for idx, point in enumerate(voxel_centers):
                self.env.engine.move_base(self.probe_voxel.object_id, point, np.array([0,0,0,1]))
                #checks if point is in close distance of the robot
                query = pyb.getClosestPoints(self.env.engine._geometry[self.probe_voxel.object_id], self.env.engine._robots[self.virtual_robot.object_id], self.robot_voxel_safe_distance)      
                delete_mask[idx] = False if query else True
            voxel_centers = voxel_centers[delete_mask]

            self.voxel_centers = voxel_centers

    def VoxelsToPybullet(self):
        if not self.running_rrt:
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 0)
            # update voxel positions
            for idx, voxel in enumerate(self.voxels):
                if idx >= len(self.voxel_centers):
                    # set all remaining voxels to nowhere
                    for i in range(idx, len(self.voxels)):
                        self.voxels[i].position = self.pos_nowhere
                        self.env.engine.move_base(self.voxels[i].object_id, self.pos_nowhere, np.array([0,0,0,1]))
                    break
                self.env.engine.move_base(voxel.object_id, self.voxel_centers[idx], np.array([0,0,0,1]))              
                voxel.position = self.voxel_centers[idx]
            # calculate new colors
            if self.color_voxels:
                voxel_norms = np.linalg.norm(self.voxel_centers - self.camera_transform_to_pyb_origin[:3, 3], axis=1)
                max_norm, min_norm = np.max(voxel_norms), np.min(voxel_norms)
                voxel_norms = (voxel_norms - min_norm) / (max_norm -  min_norm)
                colors = np.ones((len(voxel_norms), 4))
                colors = np.multiply(colors, voxel_norms.reshape(len(voxel_norms),1))
                colors[:,3] = 1
                colors[:,2] *= 0.5
                colors[:,1] *= 0.333
                colors[:,:3] = 1 - colors[:, :3]
                for idx, voxel in enumerate(self.voxels):
                    if idx >= len(self.voxel_centers):
                        break
                    pyb.changeVisualShape(self.env.engine._geometry[voxel.object_id], -1, rgbaColor=colors[idx])
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_RENDERING, 1)
            
    
    def delete_points_by_circle_center(self, points, pos):
        # returns Boolean np.array
        probe_norm = np.linalg.norm((points - pos), axis=1)
        return probe_norm > self.robot_voxel_safe_distance

    

    #liest aktuelle Joints des real ur5 aus und überträgt sie in die Simulation
    def cbSimSync(self, event):
        if self.joints is not None and not self.running_rrt:  # only move if joints are not None symsinc wird nur aufgerufen, wenn RRT aktuell nicht läuft
            self.virtual_robot.moveto_joints(self.joints, False) #False = dont use physics sim 
        self.virtual_robot.position_rotation_sensor.update(0)
        self.end_effector_xyz_sim = self.virtual_robot.position_rotation_sensor.position

    # holen Daten, Frequenz: vom Sender (UR5)
    def cbGetPos(self, data):
        for entry in data.transforms:
            parent_frame_id = entry.header.frame_id 
            frame_id = entry.child_frame_id
            if parent_frame_id == "base" and frame_id == "tool0_controller":
                xyz = np.array([entry.transform.translation.x, entry.transform.translation.y, entry.transform.translation.z])
                quat = np.array([entry.transform.rotation.x, entry.transform.rotation.y, entry.transform.rotation.z, entry.transform.rotation.w])
                self.end_effector_xyz = xyz
                self.end_effector_quat = -quat #anpassung an rviz
                self.end_effector_rpy = pyb.getEulerFromQuaternion(self.end_effector_quat)
                
            
    def cbGetJoints(self, data):
        # sometimes the robot driver reports joint angles in an order different from the one we need
        # that's why we need to map the reported angles to our order that is given in JOINT_NAMES
        orig_data = dict()
        for idx, name in enumerate(data.name):
            orig_data[name] = data.position[idx]
            
        output = []
        for name in JOINT_NAMES:
            output.append(orig_data[name])

        self.joints = np.array(output, dtype=np.float32)
        

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener = listener_node_one(action_rate=2, control_rate=4, num_voxels=2000, point_cloud_static=True)
