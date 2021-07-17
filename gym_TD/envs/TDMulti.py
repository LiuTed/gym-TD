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
        if hyper_parameters.allow_multiple_actions:
            self.action_space = spaces.Dict({
                "Attacker": spaces.MultiBinary(3),
                "Defender": spaces.Box(low=0., high=2., shape=(4, map_size, map_size), dtype=np.int32)
            })
        else:
            self.action_space = spaces.Dict({
                "Attacker": spaces.Discrete(4),
                "Defender": spaces.Discrete(map_size*map_size*4+1)
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
        real_act = {}
        atk_act = action["Attacker"]
        def_act = action["Defender"]
        if hyper_parameters.allow_multiple_actions:
            real_act["Attacker"] = []
            if self.attacker_cd == 0:
                for i in range(3):
                    if atk_act[i] == 1:
                        res = self.__board.summon_enemy(i)
                        if res:
                            self.attacker_cd = config.attacker_action_interval
                            real_act["Attacker"].append(i)

            real_act["Defender"] = np.zeros(shape=(4, self.__board.map_size, self.__board.map_size), dtype=np.int32)
            if self.defender_cd == 0:
                for r in range(self.__board.map_size):
                    for c in range(self.__board.map_size):
                        if def_act[0][r][c] == 1:
                            res = self.__board.tower_build([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][0, r, c] = 1
                        if def_act[1][r][c] == 1:
                            res = self.__board.tower_atkup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][1, r, c] = 1
                        if def_act[2][r][c] == 1:
                            res = self.__board.tower_rangeup([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][2, r, c] = 1
                        if def_act[3][r][c] == 1:
                            res = self.__board.tower_destruct([r,c])
                            if res:
                                self.defender_cd = config.defender_action_interval
                                real_act["Defender"][3, r, c] = 1
        else:
            real_act["Attacker"] = 3
            if self.attacker_cd == 0:
                if atk_act < 3:
                    res = self.__board.summon_enemy(atk_act)
                    if res:
                        self.attacker_cd = config.attacker_action_interval
                        real_act["Attacker"] = atk_act

            real_act["Defender"] = self.map_size*self.map_size*4
            if self.defender_cd == 0 and def_act != self.map_size*self.map_size*4:
                act = def_act // (self.map_size*self.map_size)
                r = (def_act // self.map_size) % self.map_size
                c = def_act % self.map_size
                if act == 0:
                    res = self.__board.tower_build([r, c])
                elif act == 1:
                    res = self.__board.tower_atkup([r, c])
                elif act == 2:
                    res = self.__board.tower_rangeup([r, c])
                elif act == 3:
                    res = self.__board.tower_destruct([r, c])
                if res:
                    self.defender_cd = config.defender_action_interval
                    real_act = def_act
            
        reward = self.__board.step()
        done = self.__board.done()
        states = self.__board.get_states()
        return states, reward, done, {"RealAction": real_act}

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

