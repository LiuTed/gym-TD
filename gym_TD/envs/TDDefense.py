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
        if hyper_parameters.allow_multiple_actions:
            self.action_space = spaces.Box(low=0., high=2., shape=(config.tower_types+2, map_size, map_size), dtype=np.int64)
        else:
            self.action_space = spaces.Discrete(map_size*map_size*(config.tower_types+2)+1)
        self.difficulty = difficulty
        self.name = "TDDefense"
    
    def empty_action(self):
        if hyper_parameters.allow_multiple_actions:
            return np.zeros((config.tower_types+2, self._board.map_size, self._board.map_size), dtype=np.int64)
        else:
            return self._board.map_size*self._board.map_size*(config.tower_types+2)
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        if hyper_parameters.allow_multiple_actions:
            real_act = np.zeros(shape=(config.tower_types+2, self._board.map_size, self._board.map_size), dtype=np.int64)
            if self.defender_cd == 0:
                for r in range(self._board.map_size):
                    for c in range(self._board.map_size):
                        for t in range(config.tower_types):
                            if action[t][r][c] == 1:
                                res = self._board.tower_build(t, [r,c])
                                if res:
                                    self.defender_cd = config.defender_action_interval
                                    real_act[t, r, c] = 1
                        if action[config.tower_types][r][c] == 1:
                            res = self._board.tower_lvup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[config.tower_types, r, c] = 1
                        if action[config.tower_types+1][r][c] == 1:
                            res = self._board.tower_destruct([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[config.tower_types+1, r, c] = 1
        else:
            fail_code = 0
            real_act = self.map_size*self.map_size*6
            if self.defender_cd == 0 and action != self.map_size*self.map_size*(config.tower_types+2):
                act = action // (self.map_size*self.map_size)
                r = (action // self.map_size) % self.map_size
                c = action % self.map_size
                if act < config.tower_types:
                    res = self._board.tower_build(act, [r, c])
                elif act == config.tower_types:
                    res = self._board.tower_lvup([r, c])
                elif act == config.tower_types+1:
                    res = self._board.tower_destruct([r, c])
                if res:
                    self.defender_cd = config.defender_action_interval
                    real_act = action
                fail_code = self._board.fail_code
        
        getattr(self, 'random_enemy_lv{}'.format(self.difficulty))()
        
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
            'AllowNextMove': self.defender_cd <= 1,
            'FailCode': fail_code,
            'ValidMask': vm
        }
        return states, reward, done, info
