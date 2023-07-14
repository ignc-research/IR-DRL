# Deployment

## Introduction
In order to guarantee the practical applicability of our agents beyond simulations, we have developed our Sim2Real method. This approach, in combination with our perception pipeline, has significantly reduced the Simulation-to-Real Gap. As a result, all agents trained within our virtual environment are now deployable in changing real-world scenarios. These agents possess the capability to respond to dynamic obstacles and human interventions, as demonstrated in our selected examples.

## Setup 

- Camera: [Intel-Realsense-Depth-Camera-D435i](https://www.intelrealsense.com/depth-camera-d435i/)
- Ros Version: [Ros-Noetic](http://wiki.ros.org/noetic)
- Camera ROS SDK : [Realsense-Ros](https://github.com/IntelRealSense/realsense-ros)
- Ros UR5 Drivers: [UR5-ROS1](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
- Voxelization: [Open-3D](http://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html#)
- Robot: [Universal-Robots-UR5](https://www.universal-robots.com/de/produkte/ur5-roboter/?utm_source=Google&utm_medium=cpc&utm_cja=Demo&utm_leadsource=Paid%20Search&utm_campaign=HQ_DE_Always-On2021&utm_content=textad&utm_term=ur5&gad=1&gclid=CjwKCAjw5MOlBhBTEiwAAJ8e1jz5oA_BB14kaMQqndz8n05QTx-iGjem9-KQcpxc-RoL6T_Q3ZqIxhoCEDgQAvD_BwE)


## Getting Started 

To begin, you'll need to install the ROS1 UR5 Drivers ([Ros-Noetic](http://wiki.ros.org/noetic)). We recommend performing this installation on a native Ubuntu 20.04 system, as virtual machines have been known to cause issues. Following this, install the ROS-SDK for your depth camera, such as Realsense-Ros for Intel Realsense cameras([Realsense-Ros](https://github.com/IntelRealSense/realsense-ros)). .

Our system utilizes an *Intel Realsense D435i*, but the method is compatible with any depth camera that supports ROS Noetic. If you're using an Intel Realsense camera, you can use our configuration file located in the Sim2Real directory at Sim2Real/config_data/config.yaml. If you're using a different camera, you'll need to set the "Pointcloud_ROS_Path" variable to the name of the rostopic that provides the point cloud.

We recommend installing our modular_drl_env as a pip package. This allows you to run our Sim2Real files from any directory. To do this, navigate to the main IR-DRL directory and execute the following commands:

We recommend installing our *modular_drl_env* as a pip package, so that you can run our Sim2Real files from any directory. To do that you need to run 
 ```python setup.py sdist ``` and afterwards ``` pip install dist ``` in our main IR-DRL directory. 

Once the installation is complete, you should be able to execute our Sim2Real Method by running the move_DRL_main.py file. Please ensure that the camera and robot are properly connected before attempting to run the file.

 A more detailled instruction will be available in coding guide (coming soon)