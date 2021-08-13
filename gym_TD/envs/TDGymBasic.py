import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDBoard import TDBoard
from gym_TD import logger

import numpy as np

class TDGymBasic(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed):
        super(TDGymBasic, self).__init__()
        self.observation_space = \
            spaces.Box(low=0., high=2., shape=(map_size, map_size, TDBoard.n_channels()), dtype=np.float32)
        self.map_size = map_size
        self._board = None
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        if self._board is not None:
            self._board.close()
        self.num_roads = self.np_random.randint(low=1, high=hyper_parameters.max_num_of_roads+1)
        self._board = TDBoard(
            self.map_size,
            self.num_roads,
            self.np_random,
            config.defender_init_cost,
            config.attacker_init_cost,
            config.max_cost,
            config.base_LP
        )
        self.attacker_cd = 0
        self.defender_cd = 0
        states = self._board.get_states()
        return states
    
    def test(self):
        self._board.summon_cluster([0, 1, 2], 0)
        self._board.tower_build([self.map_size//2, self.map_size//2])
        self._board.step()
    
    def random_enemy(self):
        if self.attacker_cd == 0:
            cluster = self.np_random.randint(0, 4, [hyper_parameters.max_cluster_length], dtype=np.int64)
            road = self.np_random.randint(self.num_roads)
            if self._board.summon_cluster(
                cluster, road
            ):
                self.attacker_cd = config.defender_action_interval
                
    def random_tower_lv0(self):
        if self.defender_cd == 0:
            r, c = self.np_random.randint(0, self.map_size, [2,])
            if self._board.tower_build([r,c]):
                self.defender_cd = config.defender_action_interval

    def random_tower_lv1(self):
        if self.defender_cd == 0:
            act = self.np_random.randint(0, 3)
            if act == 0:
                for r in range(self.map_size):
                    for c in range(self.map_size):
                        if self._board.map[r, c, 0] == 1:
                            pos = [
                                [r-1, c-1],
                                [r-1, c+1],
                                [r+1, c-1],
                                [r+1, c+1]
                            ][self.np_random.randint(4)]
                            if self._board.is_valid_pos(pos) and self._board.tower_build(pos):
                                self.defender_cd = config.defender_action_interval
                                return
            elif act == 1:
                if len(self._board.towers) == 0:
                    return
                id = self.np_random.randint(0, len(self._board.towers))
                if self._board.tower_atkup(self._board.towers[id].loc):
                    self.defender_cd = config.defender_action_interval
                    return
            elif act == 2:
                if len(self._board.towers) == 0:
                    return
                id = self.np_random.randint(0, len(self._board.towers))
                if self._board.tower_rangeup(self._board.towers[id].loc):
                    self.defender_cd = config.defender_action_interval
                    return
            else:
                return

    def render(self, mode="human"):
        return self._board.render(mode)

