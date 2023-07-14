# Setup

ModularDRLEnv is built on Python 3.10 and depends on a number of packages:
- numpy
- stable_baselines3
- sb3-contrib
- torch
- pybullet
- pandas
- tensorboard
- pyaml
- pybullet-planning

Running ```pip install -r requirements.txt``` will install these in versions that are known to work with the repo. We recommend working with a conda env, in which case you can run ```conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``` to ensure that GPU-support is present for training and inference.

# Issues

On some OS distributions one of the following problems might occur:

- Sometimes the installation of OpenAI Gym might fail during building its wheel. This can be fixed by running ```pip install wheel==0.38.4 setuptools==65.5.1```
- On Linux, the package ghalton, a dependency of pybullet-planning, might fail during installation. In order to fix this, run ```pip install ghalton==0.6.1```