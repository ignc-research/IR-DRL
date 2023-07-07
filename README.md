# Modular DLR Gym Env for Robots with PyBullet


<p float="left">
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/resources/gifs/GifReal.gif" width="400" />
  <img src="https://github.com/ignc-research/IR-DRL/blob/sim2real/resources/gifs/GifSim.gif" width="400" /> 
</p>


## Introduction

This repository provides a platform for training virtual agents in robotics tasks using Deep Reinforcement Learning (DRL). The code is built on OpenAI Gym, Stable Baselines, and PyBullet. The system is designed to operate using a modular approach, giving the user the freedom to combine, configure, and repurpose single components like goal, robot or sensor types.

An integral part of this project is the implementation of a transition mechanism from a simulated to a real-world environment. By leveraging the functionalty of ROS (Robot Operating System) and Voxelisation techniques with Open 3D, there is a system established that can effectively deploy trained models into real-world scenarios. There they are able to deal with static and dynamic obstacles.

This project is intended to serve as a resource for researchers, robotics enthusiasts, and professional developers interested in the application of Deep Reinforcement Learning in robotics.

## Getting Started

To get started with the project, please follow the instructions in the following sections:

- [Setup](./Setup.md): Instructions for setting up and installing the project.
- [Training/Evaluation](./Training_Evaluation.md): Information on how to train and evaluate the models.
- [Perception](./Perception.md): Details about the perception module of the project.
- [Deployment](./Deployment.md): Guidelines for deploying the project in a Real World environment.

Please ensure you read through all the sections to understand how to use the project effectively.