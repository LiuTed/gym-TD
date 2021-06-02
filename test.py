from os import wait
import gym
from gym import logger, wrappers
from gym_TD.envs.TDSingle import TDSingle

from time import sleep

env = gym.make('TD-def-middle-v0')
env = wrappers.Monitor(env, directory='/tmp', force=True)
env.reset()
env.test()
env.render()
while True:
    sleep(1)
    env.empty_step()
    env.render()
    pass
