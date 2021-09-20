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

    def __init__(self, map_size, difficulty = 1, seed = None, fixed_seed = False, random_agent = True):
        super(TDAttack, self).__init__(map_size, seed, fixed_seed, random_agent)
        self.action_space = spaces.Box(low=0, high=config.enemy_types, shape=(hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), dtype=np.int64)
        self.difficulty = difficulty
        self.name = "TDAttack"

    def empty_action(self):
        return np.full((hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), config.enemy_types)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        
        real_act = np.copy(action)
        fail_code = []
        if self.attacker_cd == 0:
            for i in range(self.num_roads):
                cluster = action[i]
                if np.all(cluster == config.enemy_types):
                    fail_code.append(0)
                    continue
                res, real = self._board.summon_cluster(cluster, i)
                if res:
                    self.attacker_cd = config.attacker_action_interval
                real_act[i] = real
                fail_code.append(self._board.fail_code)
        
        getattr(self, 'random_tower_lv{}'.format(self.difficulty))()

        reward = -self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        win = None
        if done:
            win = self._board.base_LP is None or self._board.base_LP <= 0
        return states, reward, done, {'RealAction': real_act, 'Win': win, 'AllowNextMove': self.attacker_cd <= 1, 'FailCode': fail_code}
