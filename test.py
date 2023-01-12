from gym_env.environment import ModularDRLEnv
from time import sleep

testo = ModularDRLEnv({})
testo.reset()
print(testo.observation_space)
print(testo.action_space)

while True:
    testo.step(None)
    sleep(0.005)