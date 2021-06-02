from os import wait
import gym
from gym import logger, wrappers
from gym_TD.envs.TDSingle import TDSingle

from time import sleep

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--random', action='store_true', help='play a random game')
    args = parser.parse_args()

    env = gym.make('TD-def-middle-v0')
    env.reset()
    env.render()
    if args.random:
        while True:
            sleep(1)
            env.random_enemy()
            env.step(env.action_space.sample())
            # random actions usually build and destruct tower in a single step
            # you could comment out the envs/TDSingle.py:48-49 for better experience
            env.render()
    else:
        env.test()
        env.render()
        while True:
            sleep(1)
            env.random_enemy()
            env.empty_step()
            env.render()


