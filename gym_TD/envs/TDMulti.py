import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDBoard import TDBoard

import numpy as np

class TDMulti(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
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
        self.observation_space = \
            spaces.Box(low=0., high=2., shape=(map_size, map_size, TDBoard.n_channels()), dtype=np.float32)
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

        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        if self.attacker_cd == 0:
            for i in range(3):
                if action["Attacker"][i] == 1:
                    res = self.__board.summon_enemy(i)
                    if res:
                        self.attacker_cd = config.attacker_action_interval
                        if not config.allow_multiple_action:
                            break
        
        if self.defender_cd == 0:
            def_act = action["Defender"]
            stop = False
            for r in range(self.__board.map_size):
                for c in range(self.__board.map_size):
                    if def_act["Build"][r][c] == 1:
                        res = self.__board.tower_build([r,c])
                        if res:
                            self.defender_cd = config.defender_action_interval
                            if not config.allow_multiple_action:
                                stop = True
                                break
                    if def_act["ATKUp"][r][c] == 1:
                        res = self.__board.tower_atkup([r,c])
                        if res:
                            self.defender_cd = config.defender_action_interval
                            if not config.allow_multiple_action:
                                stop = True
                                break
                    if def_act["RangeUp"][r][c] == 1:
                        res = self.__board.tower_rangeup([r,c])
                        if res:
                            self.defender_cd = config.defender_action_interval
                            if not config.allow_multiple_action:
                                stop = True
                                break
                    if def_act["Destruct"][r][c] == 1:
                        res = self.__board.tower_destruct([r,c])
                        if res:
                            self.defender_cd = config.defender_action_interval
                            if not config.allow_multiple_action:
                                stop = True
                                break
                if stop:
                    break
        reward = self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
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
        self.attacker_cd = 0
        self.defender_cd = 0
        states = self.__board.get_states()
        return states
    
    def test(self):
        self.__board.summon_enemy(0)
        self.__board.summon_enemy(1)
        self.__board.summon_enemy(2)
        self.__board.tower_build([self.map_size//2, self.map_size//2])
        self.__board.step()
    def empty_step(self):
        self.attacker_cd = max(self.attacker_cd-1, 0)
        self.defender_cd = max(self.defender_cd-1, 0)
        reward = self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
        return states, reward, done, {}
    
    def random_enemy(self):
        if self.attacker_cd == 0:
            t = self.np_random.randint(0, 4)
            if t == 3:
                return
            if self.__board.summon_enemy(t):
                self.attacker_cd = config.defender_action_interval
    def random_tower(self):
        if self.defender_cd == 0:
            r, c = self.np_random.randint(0, self.map_size, 2)
            if self.__board.tower_build([r,c]):
                self.defender_cd = config.defender_action_interval

    def render(self, mode="human"):
        return self.__board.render(mode)

