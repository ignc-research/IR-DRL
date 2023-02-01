# -*- encoding: utf-8 -*-

#This is to generate random Waypoints to simulate a stream of data 
#TODO: duration is hardcoded, maybe it makes sense to make it dynamic

import random
import yaml

def generate_waypoint():
    #file_path = '/home/marktschreier/Desktop/Real-Life-Deployment/waypointdata/waypoints.yaml'
    file_path = '/home/marktschreier/catkin_ws/src/waypointdata/waypoints.yaml'
    duration_path = '/home/marktschreier/catkin_ws/src/waypointdata/duration.yaml'
    waypoint_list = []
    duration_list=[]
    x = round((-2 + (random.random()*2))*100)/100
    y = round((-2 + (random.random()*2))*100)/100
    z = round((-2 + (random.random()*2))*100)/100
    
    waypoint = [x,y,z,0,0,0]
    print(waypoint)
    waypoint_list.append(waypoint)
    duration_list.append(5.0)
    with open(file_path,'w') as f:
        yaml.dump(waypoint_list,f,default_flow_style=False)
        with open(duration_path,'w') as f:
            yaml.dump(duration_list,f,default_flow_style=False)
    #need to use raw_input insteaf of input because of python2
    


if __name__ == "__main__":
    generate_waypoint()