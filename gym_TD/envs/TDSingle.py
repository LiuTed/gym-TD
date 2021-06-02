import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config
from gym_TD.envs.TDBoard import TDBoard

import numpy as np

class TDSingle(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frams_per_second': 10
    }

    def __init__(self, map_size):
        super(TDSingle, self).__init__()
        self.action_space = spaces.Dict({
            "Build": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.bool),
            "ATKUp": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.bool),
            "RangeUp": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.bool),
            "Destruct": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.bool)
        })
        self.observation_space = spaces.Dict({
            "Map": spaces.Box(low=0., high=1., shape=(map_size, map_size, 9), dtype=np.float32),
            "Cost": spaces.Discrete(100)
        })
        self.map_size = map_size
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        for r in range(self.__board.map_size):
            for c in range(self.__board.map_size):
                if action["Build"][r][c]:
                    self.__board.tower_build([r,c])
                if action["ATKUp"][r][c]:
                    self.__board.tower_atkup([r,c])
                if action["RangeUp"][r][c]:
                    self.__board.tower_rangeup([r,c])
                if action["Destruct"][r][c]:
                    self.__board.tower_destruct([r,c])
        reward = self.__board.step()
        done = self.__board.steps < 500
        states = {
            "Map": self.__board.get_states(),
            "Cost": self.__board.cost_def
        }
        return states, reward, done, None

    def reset(self):
        self.__board = TDBoard(
            self.map_size,
            self.np_random,
            config.defender_init_cost,
            config.attacker_init_cost,
            50
        )
        return self.__board.get_states()
    
    def test(self):
        self.__board.summon_enemy(0)
        self.__board.summon_enemy(1)
        self.__board.summon_enemy(2)
        self.__board.tower_build([self.map_size//2, self.map_size//2])
        self.__board.step()
    def empty_step(self):
        self.__board.step()

    def render(self, mode="human"):
        return self.__board.render(mode)

