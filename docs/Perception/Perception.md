# Perception

<div align="center">
  <img src="https://github.com/ignc-research/IR-DRL/blob/readme_overhaul/docs/Perception/gifs/Perception.gif.gif" width="400" />
</div>

## Introduction

For the purpose of enabling object detection in both dynamic and static scenarios, we implement a voxelisation process on the point cloud data captured by our depth camera. This approach allows our agents to adapt effectively to various static environments. Furthermore, it provides the capability to respond to human intervention and dynamically appearing obstacles. This setup ensures a robust performance of our agents in diverse and changing conditions.

## Setup

- Camera: [Intel-Realsense-Depth-Camera-D435i](https://www.intelrealsense.com/depth-camera-d435i/)
- Ros Version: [Ros-Noetic](http://wiki.ros.org/noetic)
- Camera ROS SDK : [Realsense-Ros](https://github.com/IntelRealSense/realsense-ros)
- Voxelization: [Open-3D](http://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html#)

For those interested in replicating our Perception Pipeline, we have provided a comprehensive tutorial in the [Deployment](docs/Deployment.md) section. We encourage you to refer to this guide for detailed instructions and insights.

## Clustering
In order to prevent the issue of 'free floating voxels'— anomalies occasionally misidentified by the camera— we use a clustering method grounded in K-D Trees. To ensure minimal latency relative to real-time movements, we have implemented a parallelized approach to the nearest neighbour search.

TODO: Pictures of Before Clustering, after Clustering


## Voxel Size and Clustering Adjustemens
The precision and performance of the system can be significantly influenced by the voxel size and the number of voxels. By default, our configuration utilizes a *voxel size* of *0.1* and a *clustering size* of *5*. 

However, for more performance-oriented setups, a *voxel size* of *0.035* and a *clustering size* of *50* may be more suitable. We encourage users to experiment with these parameters to identify the configuration that best suits their system's capabilities and their specific requirements.


<div>
  <figure style="display: inline-block">
    <img src="https://github.com/ignc-research/IR-DRL/blob/readme_overhaul/docs/Perception/gifs/voxelfein.png" width="400" />
    <figcaption>small Voxel size and high number of voxels</figcaption>
  </figure>
  <figure style="display: inline-block">
    <img src="https://github.com/ignc-research/IR-DRL/blob/readme_overhaul/docs/Perception/gifs/voxelgrob1.png" width="400" />
    <figcaption>big Voxel size and low number of voxels</figcaption>
  </figure>
</div>