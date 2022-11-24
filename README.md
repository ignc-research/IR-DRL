# IR-DRL
## Dependency
- numpy
- pybullet
- gym
- stable_baselines3
- tensorflow 2

## ur5
### Introduction
- ur5_description: robot model
- StaticEnv: environment generates static obstacles
- DynamicEnv: environment generates moving obstacles
- SimpleMixEnv: combine both, but won't happen at the same time
- [...]ImproveEnv: add more rays on the wirst

### Run
just run train.py and test.py in each env

## kuka_shelf
### Introduction
- kuka16_description: robot model
- kuka_avoid_obstacles_v1: the environment of crossing the shelf to reach the target
- kuka_avoid_obstacles_v2: add the shaking penalty in reward function compared with v1

### Run
set IS_TRAIN in reach_train_ppo.py for switching training or testing
