# Modular DLR Gym Env for Robots with PyBullet

This repo implements a modular Gym env in which all relevant components, i.e. robots, obstacles, sensors and training goals, can be switched out at will without much or any change in the code.

## Dependencies

- Python 3.10
- numpy
- gym
- stable_baselines3
- sb3-contrib
- torch
- pybullet
- pandas
- tensorboard
- zennit
- mazelib

Once python is installed, you can use ```pip install -r requirements.txt``` to install the above packages in versions that work with the code in this repo.

## State of the code & plans

The gym env is now fully modular and can be controlled by pre-written YAML config files.

To implement your own sensors, scenarios, robots or goals refer to the existing implementations. Both the abstract base classes and the implementations feature explicit instructions and commentary for writing your own version.

Implemented:
- Robots:
    - movement via Inverse Kinematics, Joint Angles and Joint Velocities
    - UR5
    - Kuka KR16
- Sensors:
    - Position & Rotation
    - Joints
    - Lidar
    - Camera (floating, floating & following, mounted on EE)
- Worlds (Scenarios):
    - Random Obstacles (Yifan)
    - Table Experiment with humans and random obstacles
- Goal:
    - Position goal with collision avoidance (Yifan)

Coming soon:
- the three old testcases (one plate, two plates, moving plate)
- variations of the testcases (different directions and angles)

## Running the code

To start training or evaluation, run ```python run.py configfile --train|--eval```. configfile should be the path to a valid YAML config file (see the examples in /configs/) while --train and --eval determine the mode the env will run in.