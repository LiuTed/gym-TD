import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDBoard import TDBoard

import numpy as np

import random

class TDAttack(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size):
        super(TDAttack, self).__init__()
        if hyper_parameters.allow_multiple_actions:
            self.action_space = spaces.MultiBinary(3)
        else:
            self.action_space = spaces.Discrete(4)
        self.observation_space = \
            spaces.Box(low=0., high=2., shape=(map_size, map_size, TDBoard.n_channels()), dtype=np.float32)
        self.map_size = map_size
        self.__board = None
        self.seed()
        self.reset()
        self.attacker_cd = 0
        self.defender_cd = 0
        self.last_act = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        if hyper_parameters.allow_multiple_actions:
            real_act = []
            if self.attacker_cd == 0:
                for i in range(3):
                    if action[i] == 1:
                        res = self.__board.summon_enemy(i)
                        if res:
                            self.attacker_cd = config.attacker_action_interval
                            real_act.append(i)
        else:
            real_act = 3
            if self.attacker_cd == 0:
                if action < 3:
                    res = self.__board.summon_enemy(action)
                    if res:
                        self.attacker_cd = config.attacker_action_interval
                        real_act = action
        
        # self.random_tower_lv0()
        self.random_tower_lv1()

        reward = -self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
        return states, reward, done, {'RealAction': real_act}

    def reset(self):
        if self.__board is not None:
            self.__board.close()
        self.__board = TDBoard(
            self.map_size,
            self.np_random,
            config.defender_init_cost,
            config.attacker_init_cost,
            config.max_cost,
            config.base_LP
        )
        self.attacker_cd = 0
        self.defender_cd = 0
        states = self.__board.get_states()
        return states
    
    def test(self):
        self.__board.summon_enemy(0)
        self.__board.summon_enemy(1)
        self.__board.summon_enemy(2)
        self.__board.tower_build([self.map_size//2, self.map_size//2])
        self.__board.step()
    def empty_step(self):
        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        reward = -self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
        return states, reward, done, {}
    def random_tower_lv0(self):
        if self.defender_cd == 0:
            r, c = self.np_random.randint(0, self.map_size, 2)
            if self.__board.tower_build([r,c]):
                self.defender_cd = config.defender_action_interval
    def random_tower_lv1(self):
        if self.defender_cd == 0:
            act = random.randint(0, 3)
            if act == 0:
                for r in range(self.map_size):
                    for c in range(self.map_size):
                        if self.__board.map[r, c, 0] == 1:
                            pos = random.choice([
                                [r-1, c-1],
                                [r-1, c+1],
                                [r+1, c-1],
                                [r+1, c+1]
                            ])
                            if self.__board.is_valid_pos(pos) and self.__board.tower_build(pos):
                                self.defender_cd = config.defender_action_interval
                                return
            elif act == 1:
                if len(self.__board.towers) == 0:
                    return
                id = random.randint(0, len(self.__board.towers)-1)
                if self.__board.tower_atkup(self.__board.towers[id].loc):
                    self.defender_cd = config.defender_action_interval
                    return
            elif act == 2:
                if len(self.__board.towers) == 0:
                    return
                id = random.randint(0, len(self.__board.towers)-1)
                if self.__board.tower_rangeup(self.__board.towers[id].loc):
                    self.defender_cd = config.defender_action_interval
                    return
            else:
                return


    def render(self, mode="human"):
        return self.__board.render(mode)

