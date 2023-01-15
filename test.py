from gym_env.environment import ModularDRLEnv
from time import sleep

testo = ModularDRLEnv({})
testo.reset()

while True:
    obs, reward, done, info = testo.step(testo.action_space.sample())
    sleep(0.005)