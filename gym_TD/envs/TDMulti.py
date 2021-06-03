import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config
from gym_TD.envs.TDBoard import TDBoard

import numpy as np

class TDMulti(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frams_per_second': 10
    }

    def __init__(self, map_size):
        super(TDMulti, self).__init__()
        self.action_space = spaces.Dict({
            "Attacker": spaces.MultiBinary(3),
            "Defender": spaces.Dict({
                "Build": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.int32),
                "ATKUp": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.int32),
                "RangeUp": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.int32),
                "Destruct": spaces.Box(low=0, high=1, shape=(map_size, map_size), dtype=np.int32)
            })
        })
        self.observation_space = spaces.Dict({
            "Map": spaces.Box(low=0., high=1., shape=(map_size, map_size, 9), dtype=np.float32),
            "Cost_Attacker": spaces.Discrete(config.max_cost),
            "Cost_Defender": spaces.Discrete(config.max_cost)
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

        for i in range(3):
            if action["Attacker"][i] == 1:
                self.__board.summon_enemy(i)
        def_act = action["Defender"]
        for r in range(self.__board.map_size):
            for c in range(self.__board.map_size):
                if def_act["Build"][r][c] == 1:
                    self.__board.tower_build([r,c])
                if def_act["ATKUp"][r][c] == 1:
                    self.__board.tower_atkup([r,c])
                if def_act["RangeUp"][r][c] == 1:
                    self.__board.tower_rangeup([r,c])
                if def_act["Destruct"][r][c] == 1:
                    self.__board.tower_destruct([r,c])
        reward = self.__board.step()
        done = self.__board.steps >= 500
        states = {
            "Map": self.__board.get_states(),
            "Cost_Defender": self.__board.cost_def,
            "Cost_Attacker": self.__board.cost_atk
        }
        return states, reward, done, None

    def reset(self):
        self.__board = TDBoard(
            self.map_size,
            self.np_random,
            config.defender_init_cost,
            config.attacker_init_cost,
            config.max_cost
        )
        states = {
            "Map": self.__board.get_states(),
            "Cost": self.__board.cost_def
        }
        return states
    
    def test(self):
        self.__board.summon_enemy(0)
        self.__board.summon_enemy(1)
        self.__board.summon_enemy(2)
        self.__board.tower_build([self.map_size//2, self.map_size//2])
        self.__board.step()
    def empty_step(self):
        reward = self.__board.step()
        done = self.__board.steps >= 500
        states = {
            "Map": self.__board.get_states(),
            "Cost_Defender": self.__board.cost_def,
            "Cost_Attacker": self.__board.cost_atk
        }
        return states, reward, done, None
    
    def random_enemy(self):
        t = self.np_random.randint(0, 4)
        if t == 3:
            return
        self.__board.summon_enemy(t)
    def random_tower(self):
        r, c = self.np_random.randint(0, self.map_size, 2)
        self.__board.tower_build([r,c])

    def render(self, mode="human"):
        return self.__board.render(mode)
