import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDGymBasic import TDGymBasic

from gym_TD.utils import fail_code as FC
import numpy as np

class TDMulti(TDGymBasic):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed = None, fixed_seed = False, random_agent = True):
        super(TDMulti, self).__init__(map_size, seed, fixed_seed, random_agent)
        # self.action_space = spaces.Dict({
        #     "Attacker": spaces.Box(low=0, high=4, shape=(hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), dtype=np.int64),
        #     "Defender": spaces.Discrete(map_size*map_size*6+1)
        # })
        self.action_space = spaces.Dict({
            "Attacker": spaces.MultiDiscrete([hyper_parameters.max_num_of_roads + 1, config.enemy_types + 1]),
            "Defender": spaces.MultiDiscrete([config.tower_types+3, map_size+1, map_size+1])
        })
        self.name = "TDMulti"
    
    def empty_action(self):
        # return {
        #     "Attacker": np.full((hyper_parameters.max_num_of_roads, hyper_parameters.max_cluster_length), 4, dtype=np.int64),
        #     "Defender": self._board.map_size*self._board.map_size*6
        # }
        return {
            "Attacker": np.zeros((2,), dtype=np.int64),
            "Defender": np.zeros((3,), dtype=np.int64)
        }

    def step(self, action):
        if not self.action_space.contains(action):
            if not(isinstance(action, (tuple, list, np.ndarray)) and self.action_space.contains(action[0])):
                err_msg = "%r (%s) invalid" % (action, type(action))
                assert False, err_msg
            else:
                action = action[0]

        real_act = action
        atk_act = action["Attacker"]
        def_act = action["Defender"]

        # real_act["Attacker"] = np.copy(atk_act)
        # afail = []
        # for i in range(self.num_roads):
        #     cluster = atk_act[i]
        #     if np.all(cluster == 4):
        #         afail.append(0)
        #         continue
        #     if not self._board.summon_cluster(cluster, i):
        #         real_act["Attacker"][i] = 4
        #     afail.append(self._board.fail_code)

        # dfail = 0
        # real_act["Defender"] = self.map_size*self.map_size*6
        # if def_act != self.map_size*self.map_size*6:
        #     act = def_act // (self.map_size*self.map_size)
        #     r = (def_act // self.map_size) % self.map_size
        #     c = def_act % self.map_size
        #     if act in [0,1,2,3]:
        #         res = self._board.tower_build(act, [r, c])
        #     elif act == 4:
        #         res = self._board.tower_lvup([r, c])
        #     elif act == 5:
        #         res = self._board.tower_destruct([r, c])
        #     if res:
        #         real_act = def_act
        #     dfail = self._board.fail_code

        afail = 0
        if atk_act[0] > self.num_roads:
            real_act["Attacker"] = [0, 0]
            afail = FC.INVALID_ACTION
        elif atk_act[0] > 0:
            if 0 < atk_act[1] <= config.enemy_types:
                road = atk_act[0] - 1
                t = atk_act[1] - 1
                res = self._board.summon_enemy(t, road)
                if not res:
                    real_act["Attacker"][1] = 0
                afail = self._board.fail_code
        
        dfail = 0
        act = def_act[0]
        row = def_act[1]
        col = def_act[2]
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
                real_act["Defender"] = [0, 0, 0]
            dfail = self._board.fail_code
        
            
        reward = self._board.step()
        done = self._board.done()
        states = self._board.get_states()
        win = None
        if done:
            win = {
                'Defender': self._board.base_LP is None or self._board.base_LP > 0,
                'Attacker': self._board.base_LP is None or self._board.base_LP <= 0
            }
        info = {
            "RealAction": real_act,
            'Win': win,
            'FailCode': {
                'Attacker': afail,
                'Defender': dfail
            }
        }
        return states, reward, done, info

