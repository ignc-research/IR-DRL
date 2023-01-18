# Modular DLR Gym Env for Robots with PyBullet

This repo implements a modular Gym env in which all relevant components, i.e. robots, obstacles, sensors and training goals, can be switched out at will without much or any change in the code.

## Dependencies

- numpy
- stable_baselines3
- pytorch
- pybullet
- tensorboard

## State of the code & plans

For now, the basic setup with random obstacles, targets and a UR5 robot trying to reach them is implemented and hardcoded into the \_\_init\_\_ of the gym env. In the future, it will be possible to swap robots, scenarios and sensors via a config.
However, the basic capability to do that easily is already there, it just needs to be done by hand (initializing the objects and adding them to the list of sensors, robots etc.).

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
- Worlds (Scenarios):
    - Random Obstacles (Yifan)
- Goal:
    - Position goal with collision avoidance (Yifan)

Coming soon:
- the three old testcases (one plate, two plates, moving plate)
- variations of the testcases (different directions and angles)
- camera sensor
- camera in hand sensor

## Running the code

To start training or evaluation, run ```python run.py```. What will be run is determined by a few parameters in the run.py file. They are commented and explained at the top of the file.