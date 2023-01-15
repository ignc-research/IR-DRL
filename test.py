from gym_env.environment import ModularDRLEnv
from time import sleep

env_config_train = {
    "train": True,
    "logging": 1,
    "joint_control": False,
    "normalize_observations": False,
    "normalize_rewards": False,
    "display": False,
    "display_extra": False
}

testo = ModularDRLEnv(env_config_train)
testo.reset()

while True:
    obs, reward, done, info = testo.step(testo.action_space.sample())
    sleep(0.005)