import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDGymBasic import TDGymBasic

import numpy as np

class TDMulti(TDGymBasic):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed = None):
        super(TDMulti, self).__init__(map_size, seed)
        if hyper_parameters.allow_multiple_actions:
            self.action_space = spaces.Dict({
                "Attacker": spaces.Box(low=0, high=3, shape=(hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), dtype=np.int64),
                "Defender": spaces.Box(low=0., high=2., shape=(4, map_size, map_size), dtype=np.int64)
            })
        else:
            self.action_space = spaces.Dict({
                "Attacker": spaces.Box(low=0, high=3, shape=(hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), dtype=np.int64),
                "Defender": spaces.Discrete(map_size*map_size*4+1)
            })
        self.name = "TDMulti"
    
    def empty_action(self):
        if hyper_parameters.allow_multiple_actions:
            return {
                "Attacker": np.full((hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), 3, dtype=np.int64),
                "Defender": np.zeros((4, self._board.map_size, self._board.map_size), dtype=np.int64)
            }
        else:
            return {
                "Attacker": np.full((hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), 3, dtype=np.int64),
                "Defender": self._board.map_size*self._board.map_size*4
            }
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        real_act = {}
        atk_act = action["Attacker"]
        def_act = action["Defender"]
        if hyper_parameters.allow_multiple_actions:
            real_act["Attacker"] = np.copy(atk_act)
            if self.attacker_cd == 0:
                for i in range(self.num_roads):
                    cluster = atk_act[i]
                    if self._board.summon_cluster(cluster, i):
                        self.attacker_cd = config.attacker_action_interval
                    else:
                        real_act["Attacker"][i] = 3

            real_act["Defender"] = np.zeros(shape=(4, self._board.map_size, self._board.map_size), dtype=np.int64)
            if self.defender_cd == 0:
                for r in range(self._board.map_size):
                    for c in range(self._board.map_size):
                        if def_act[0][r][c] == 1:
                            res = self._board.tower_build([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][0, r, c] = 1
                        if def_act[1][r][c] == 1:
                            res = self._board.tower_atkup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][1, r, c] = 1
                        if def_act[2][r][c] == 1:
                            res = self._board.tower_rangeup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][2, r, c] = 1
                        if def_act[3][r][c] == 1:
                            res = self._board.tower_destruct([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][3, r, c] = 1
        else:
            real_act["Attacker"] = np.copy(atk_act)
            if self.attacker_cd == 0:
                for i in range(self.num_roads):
                    cluster = atk_act[i]
                    if self._board.summon_cluster(cluster, i):
                        self.attacker_cd = config.attacker_action_interval
                    else:
                        real_act["Attacker"][i] = 3

            real_act["Defender"] = self.map_size*self.map_size*4
            if self.defender_cd == 0 and def_act != self.map_size*self.map_size*4:
                act = def_act // (self.map_size*self.map_size)
                r = (def_act // self.map_size) % self.map_size
                c = def_act % self.map_size
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
                    real_act = def_act
            
        reward = self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        win = None
        if done:
            win = {
                'Defender': self._board.base_LP is None or self._board.base_LP > 0,
                'Attacker': self._board.base_LP is None or self._board.base_LP <= 0
            }
        return states, reward, done, {"RealAction": real_act, 'Win': win, 'AllowNextMove':{'Attacker': self.attacker_cd <= 1, 'Defender': self.defender_cd <= 1}}

