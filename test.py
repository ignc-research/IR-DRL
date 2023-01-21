from gym_env.environment import ModularDRLEnv
from time import sleep

env_config_train = {
    "train": False,
    "logging": 1,
    "max_steps_per_episode": 1024,
    "max_episodes": -1,
    "sim_step": 1 / 240,
    "stat_buffer_size": 25,
    "use_physics_sim": False,
    "control_mode": 2,
    "normalize_observations": False,
    "normalize_rewards": False,
    "dist_threshold_overwrite": None,
    "display": True,
    "display_extra": True
}

testo = ModularDRLEnv(env_config_train)


while True:
    testo.reset()
    done = False
    while not done:
        #obs, reward, done, info = testo.step(testo.action_space.sample())
        obs, reward, done, info = testo.step([0,0,0,0,0,0])
        #sleep(0.005)