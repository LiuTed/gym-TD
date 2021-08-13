import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDGymBasic import TDGymBasic

import numpy as np

import random

class TDAttack(TDGymBasic):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed = None):
        super(TDAttack, self).__init__(map_size, seed)
        self.action_space = spaces.Box(low=0, high=3, shape=(hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), dtype=np.int64)
        self.name = "TDAttack"

    def empty_action(self):
        return np.full((hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), 3)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        
        real_act = np.zeros_like(action)
        if self.attacker_cd == 0:
            for i in range(self.num_roads):
                cluster = action[i]
                if self._board.summon_cluster(cluster, i):
                    real_act[i] = cluster
                    self.attacker_cd = config.attacker_action_interval
                else:
                    real_act[i] = 3
            real_act[self.num_roads:] = action[self.num_roads:]
        
        # self.random_tower_lv0()
        self.random_tower_lv1()

        reward = -self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        win = None
        if done:
            win = self._board.base_LP is None or self._board.base_LP <= 0
        return states, reward, done, {'RealAction': real_act, 'Win': win, 'AllowNextMove': self.attacker_cd <= 1}
