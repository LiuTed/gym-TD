from gym_TD.utils import fail_code
import gym
from gym import spaces, utils
from gym.utils import seeding
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import diff

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDGymBasic import TDGymBasic

import numpy as np

class TDDefense(TDGymBasic):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, difficulty = 1, seed = None, fixed_seed = False, random_agent = True):
        super(TDDefense, self).__init__(map_size, seed, fixed_seed, random_agent)
        # self.action_space = spaces.Discrete(map_size*map_size*(config.tower_types+2)+1)
        self.action_space = spaces.MultiDiscrete([config.tower_types+3, map_size+1, map_size+1])
        self.agent = getattr(self, 'random_enemy_lv{}'.format(self.difficulty))
        self.difficulty = difficulty
        self.name = "TDDefense"
    
    def empty_action(self):
        # return self._board.map_size*self._board.map_size*(config.tower_types+2)
        return np.zeros((3,), dtype=np.int64)
    
    def step(self, action):
        if not self.action_space.contains(action):
            if not(isinstance(action, (tuple, list, np.ndarray)) and self.action_space.contains(action[0])):
                err_msg = "%r (%s) invalid" % (action, type(action))
                assert False, err_msg
            else:
                action = action[0]

        # Discrete
        # fail_code = 0
        # real_act = self.map_size*self.map_size*6
        # if action != self.map_size*self.map_size*(config.tower_types+2):
        #     act = action // (self.map_size*self.map_size)
        #     r = (action // self.map_size) % self.map_size
        #     c = action % self.map_size
        #     if act < config.tower_types:
        #         res = self._board.tower_build(act, [r, c])
        #     elif act == config.tower_types:
        #         res = self._board.tower_lvup([r, c])
        #     elif act == config.tower_types+1:
        #         res = self._board.tower_destruct([r, c])
        #     if res:
        #         real_act = action
        #     fail_code = self._board.fail_code

        # MultiDiscrete
        fail_code = 0
        real_act = action
        act = action[0]
        row = action[1]
        col = action[2]
        if act > 0 and row > 0 and col > 0:
            act -= 1
            row -= 1
            col -= 1
            if act < config.tower_types:
                res = self._board.tower_build(act, [row, col])
            elif act == config.tower_types:
                res = self._board.tower_lvup([row, col])
            else:
                res = self._board.tower_destruct([row, col])
            if not res:
                real_act = [0, 0, 0]
            fail_code = self._board.fail_code
        
        self.agent()
        
        reward = self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        vm = self._board.get_def_valid_mask()
        win = None
        if done:
            win = self._board.base_LP is None or self._board.base_LP > 0
        info = {
            'RealAction': real_act,
            'Win': win,
            'FailCode': fail_code,
            'ValidMask': vm
        }
        return states, reward, done, info
