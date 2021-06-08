from os import wait
import gym
from gym import logger, wrappers
from gym_TD.envs.TDSingle import TDSingle

from time import sleep

import argparse

def play_demo():
    env = gym.make('TD-2p-middle-v0')
    env.reset()
    env.render()
    env.test()
    env.render()
    done = False
    while not done:
        sleep(.3)
        _, _, done, _ = env.empty_step()
        env.render()

def play_atk():
    env = gym.make('TD-atk-middle-v0')
    env.reset()
    env.render()
    done = False
    while not done:
        sleep(.3)
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()

def play_def():
    env = gym.make('TD-def-middle-v0')
    env.reset()
    env.render()
    done = False
    while not done:
        sleep(.3)
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()

def play_2p():
    env = gym.make('TD-2p-middle-v0')
    env.reset()
    env.render()
    done = False
    while not done:
        sleep(.3)
        env.random_tower()
        env.random_enemy()
        _, _, done, _ = env.empty_step()
        env.render()

def test():
    env = gym.make('TD-2p-v0', map_size=21)
    logger.set_level(logger.DEBUG)
    env = wrappers.Monitor(env, directory='/tmp/gym_TD', force=True)
    for __ in range(17):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='attacker with random tower')
    parser.add_argument('-d', action='store_true', help='defender with random enemy')
    parser.add_argument('-m', action='store_true', help='all random')
    parser.add_argument('-t', action='store_true', help='debug test')
    args = parser.parse_args()

    if args.a:
        play_atk()
    elif args.d:
        play_def()
    elif args.m:
        play_2p()
    elif args.t:
        test()
    else:
        play_demo()
