import gym
from gym import logger, wrappers
import gym_TD
from gym_TD import getConfig, paramConfig, hyper_parameters, getHyperParameters
import random

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
    rs = []
    el = []
    win = []
    for i in range(100):
        env.seed(4218513)
        env.reset()
        env.render()
        done = False
        tr = 0.
        s = 0
        while not done:
            _, r, done, _ = env.step(env.action_space.sample())
            env.render()
            tr += r
            s += 1
        rs.append(tr)
        el.append(s)
        if s == hyper_parameters.max_episode_steps:
            win.append(0)
        else:
            win.append(1)
    print(sum(rs)/len(rs), sum(el)/len(el), sum(win)/len(win)*100)
        

def play_def():
    env = gym.make('TD-def-middle-v0')
    rs = []
    el = []
    win = []
    for i in range(300):
        env.reset()
        # env.render()
        done = False
        tr = 0.
        s = 0
        while not done:
            _, r, done, _ = env.step(env.action_space.sample())
            # env.render()
            tr += r
            s += 1
        rs.append(tr)
        el.append(s)
        if s == hyper_parameters.max_episode_steps:
            win.append(1)
        else:
            win.append(0)
    print(sum(rs)/len(rs), sum(el)/len(el), sum(win)/len(win)*100)

def play_2p():
    seed = random.randint(0, 0xffffff)
    print(seed)
    seed = 5807770
    env = gym.make('TD-2p-middle-v0')
    env.seed(seed)
    env.reset()
    env.render()
    done = False
    while not done:
        env.random_tower_lv1()
        env.random_enemy_lv1()
        _, _, done, _ = env.step(env.empty_action())
        env.render()
    
def single_enemy(loop, i):
    env = gym.make('TD-atk-middle-v0')
    rs = []
    el = []
    win = []
    next_act = 0
    for _ in range(loop):
        env.reset()
        done = False
        tr = 0.
        s = 0
        while not done:
            if i < 3:
                next_act = i
            __, r, done, info = env.step(next_act)
            if info["RealAction"] == next_act:
                next_act = (next_act+1)%4
            tr += r
            s += 1
        rs.append(tr)
        el.append(s)
        if s == hyper_parameters.max_episode_steps:
            win.append(0)
        else:
            win.append(1)
    print(i, ':', sum(rs)/len(rs), sum(el)/len(el), sum(win)/len(win)*100)

def test():
    print(getConfig())
    print(getHyperParameters())
    for i in [0, 1, 2, 3]:
        single_enemy(300, i)
    play_atk()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='attacker with random tower')
    parser.add_argument('-d', action='store_true', help='defender with random enemy')
    parser.add_argument('-m', action='store_true', help='all random')
    parser.add_argument('-t', action='store_true', help='debug test')
    parser.add_argument('-V', '--debug', action='store_true', help='Show debug log')
    args = parser.parse_args()

    if args.debug:
        logger.set_level(logger.DEBUG)
    else:
        logger.set_level(logger.INFO)
    
    # paramConfig(
    #     base_LP=3,
    #     reward_time=0.001,
    #     enemy_balance_LP = 15,
    #     enemy_strong_LP = 25,
    #     enemy_fast_LP = 8,
    #     enemy_balance_speed = 0.4,
    #     enemy_strong_speed = 0.22,
    #     enemy_fast_speed = 0.65,
    #     tower_attack_interval = 2,
    #     dummy = None
    # )

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
