# Modular DLR Gym Env for Robots with PyBullet

This repo implements a modular DRL Gym env for training policies for robots. Major components are kept as abstracts, enabling easy reconfiguration of all relevant objects, e.g. robots, sensors or the scenario.

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
Due to a problem with a dependency being tethered to an old numpy version that breaks our code it is necessary to run ```pip install numpy==1.24.1``` afterwards.

## State of the code & plans

The gym env is fully modular and can be controlled by YAML config files. Customizable defaults and examples for several scenarios can be found in the config folder. The explanations.yaml file contains an overview of all possible options with comments.

To implement your own sensors, scenarios, robots or goals refer to the existing implementations. Both the abstract base classes and the implementations feature explicit instructions and commentary for writing your own version.

Implemented:
- Engines:
    - Pybullet
    - (WIP) Isaac
- Robots:
    - movement via Inverse Kinematics, Joint Angles and Joint Velocities
    - UR5
    - Kuka KR16
    - Kuka LBR iiwa
- Sensors:
    - Position & Rotation
    - Joints
    - Lidar
    - Camera (floating, floating & following, mounted on EE)
- Worlds (Scenarios):
    - Random Obstacles
    - Table Experiment with humans and random obstacles
    - Kuka Shelf experiment
    - World Generator
    - Test Cases
- Goal:
    - Position goal with collision avoidance

## Running the code

To start training or evaluation, run ```python run.py configfile --train|--eval|--debug```. configfile should be the path to a valid YAML config file (see the examples in /configs/) while --train, --eval and --debug determine the mode the env will run in.

## pip install

If you need the code as a package elsewhere, the setup.py file in this repo will create an installable version including all asset files.