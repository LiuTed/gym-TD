import gym
from gym import spaces, utils
from gym.utils import seeding
from numpy.core.fromnumeric import shape

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDGymBasic import TDGymBasic

import numpy as np

class TDDefense(TDGymBasic):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed = None):
        super(TDDefense, self).__init__(map_size, seed)
        if hyper_parameters.allow_multiple_actions:
            self.action_space = spaces.Box(low=0., high=2., shape=(4, map_size, map_size), dtype=np.int64)
        else:
            self.action_space = spaces.Discrete(map_size*map_size*4+1)
        self.name = "TDDefense"
    
    def empty_action(self):
        if hyper_parameters.allow_multiple_actions:
            return np.zeros((4, self._board.map_size, self._board.map_size), dtype=np.int64)
        else:
            return self._board.map_size*self._board.map_size*4
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        if hyper_parameters.allow_multiple_actions:
            real_act = np.zeros(shape=(4, self._board.map_size, self._board.map_size), dtype=np.int64)
            if self.defender_cd == 0:
                for r in range(self._board.map_size):
                    for c in range(self._board.map_size):
                        if action[0][r][c] == 1:
                            res = self._board.tower_build([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[0, r, c] = 1
                        if action[1][r][c] == 1:
                            res = self._board.tower_atkup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[1, r, c] = 1
                        if action[2][r][c] == 1:
                            res = self._board.tower_rangeup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[2, r, c] = 1
                        if action[3][r][c] == 1:
                            res = self._board.tower_destruct([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act[3, r, c] = 1
        else:
            real_act = self.map_size*self.map_size*4
            if self.defender_cd == 0 and action != self.map_size*self.map_size*4:
                act = action // (self.map_size*self.map_size)
                r = (action // self.map_size) % self.map_size
                c = action % self.map_size
                if act == 0:
                    res = self._board.tower_build([r, c])
                elif act == 1:
                    res = self._board.tower_atkup([r, c])
                elif act == 2:
                    res = self._board.tower_rangeup([r, c])
                elif act == 3:
                    res = self._board.tower_destruct([r, c])
                if res:
                    self.defender_cd = config.defender_action_interval
                    real_act = action
                    
        self.random_enemy()
        
        reward = self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        win = None
        if done:
            win = self._board.base_LP is None or self._board.base_LP > 0
        return states, reward, done, {'RealAction': real_act, 'Win': win, 'AllowNextMove': self.defender_cd <= 1}
