from gym_env.environment import ModularDRLEnv
from time import sleep

env_config_train = {
    "train": False,
    "logging": 1,
    "use_physics_sim": False,
    "control_mode": 0,
    "normalize_observations": False,
    "normalize_rewards": False,
    "display": True,
    "display_extra": True
}

testo = ModularDRLEnv(env_config_train)


while True:
    testo.reset()
    done = False
    while not done:
        obs, reward, done, info = testo.step(testo.action_space.sample())
        #sleep(0.005)