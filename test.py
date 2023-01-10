from gym_env.environment import ModularDRLEnv
from time import sleep

testo = ModularDRLEnv({})
testo.reset()

while True:
    testo.step(None)
    sleep(0.005)