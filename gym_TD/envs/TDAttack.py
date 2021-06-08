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
        self.action_space = spaces.MultiBinary(3)
        self.observation_space = spaces.Dict({
            "Map": spaces.Box(low=0., high=1., shape=(map_size, map_size, len(TDBoard.channels)), dtype=np.float32),
            "Cost": spaces.Discrete(config.max_cost)
        })
        self.map_size = map_size
        self.__board = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        for i in range(3):
            if action[i] == 1:
                self.__board.summon_enemy(i)
        
        self.random_tower()

        reward = -self.__board.step()
        done = self.__board.done()
        states = {
            "Map": self.__board.get_states(),
            "Cost": self.__board.cost_atk
        }
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
        states = {
            "Map": self.__board.get_states(),
            "Cost": self.__board.cost_atk
        }
        return states
    
    def test(self):
        self.__board.summon_enemy(0)
        self.__board.summon_enemy(1)
        self.__board.summon_enemy(2)
        self.__board.tower_build([self.map_size//2, self.map_size//2])
        self.__board.step()
    def empty_step(self):
        reward = -self.__board.step()
        done = self.__board.done()
        states = {
            "Map": self.__board.get_states(),
            "Cost": self.__board.cost_atk
        }
        return states, reward, done, {}
    def random_tower(self):
        r, c = self.np_random.randint(0, self.map_size, 2)
        self.__board.tower_build([r,c])

    def render(self, mode="human"):
        return self.__board.render(mode)

