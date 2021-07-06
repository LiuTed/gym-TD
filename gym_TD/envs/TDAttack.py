import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDBoard import TDBoard

import numpy as np

class TDAttack(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size):
        super(TDAttack, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = \
            spaces.Box(low=0., high=2., shape=(map_size, map_size, TDBoard.n_channels()), dtype=np.float32)
        self.map_size = map_size
        self.__board = None
        self.seed()
        self.reset()
        self.attacker_cd = 0
        self.defender_cd = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        if self.attacker_cd == 0:
            if action < 3:
                res = self.__board.summon_enemy(action)
                if res:
                    self.attacker_cd = config.attacker_action_interval
            # for i in range(3):
            #     if action[i] == 1:
            #         res = self.__board.summon_enemy(i)
            #         if res:
            #             self.attacker_cd = config.attacker_action_interval
            #             if not config.allow_multiple_action:
            #                 break
        
        self.random_tower()

        reward = -self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
        return states, reward, done, {}

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
    def random_tower(self):
        if self.defender_cd == 0:
            r, c = self.np_random.randint(0, self.map_size, 2)
            if self.__board.tower_build([r,c]):
                self.defender_cd = config.defender_action_interval

    def render(self, mode="human"):
        return self.__board.render(mode)

