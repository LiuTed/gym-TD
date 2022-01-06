import gym
from gym import spaces, utils
from gym.utils import seeding

from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDBoard import TDBoard
from gym_TD import logger
from gym_TD.utils import fail_code as FC

import numpy as np

class TDGymBasic(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': hyper_parameters.video_frames_per_second
    }

    def __init__(self, map_size, seed, fixed_seed = False, random_agent = True):
        super(TDGymBasic, self).__init__()
        self.observation_space = \
            spaces.Box(low=0., high=TDBoard._high(map_size), shape=(map_size, map_size, TDBoard.n_channels()), dtype=np.float32)
        self.map_size = map_size
        self._board = None
        self.fixed_seed = fixed_seed
        self.input_seed = seed
        self.random_agent = random_agent
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
        if self.fixed_seed:
            self.seed(self.input_seed)
        self.num_roads = self.np_random.randint(low=1, high=hyper_parameters.max_num_of_roads+1)
        self._board = TDBoard(
            self.map_size,
            self.num_roads,
            self.np_random
        )
        states = self._board.get_states()
        return states
    
    def test(self):
        from gym_TD.envs.TDElements import create_enemy, create_tower
        start = self._board.start[0]
        for t in range(config.enemy_types):
            for r in [0.1, 0.5, 1]:
                e = create_enemy(t, start, self._board.map[4, start[0], start[1]], 0)
                e.LP = int(e.maxLP * r)
                self._board.enemies.append(e)
        self._board.cost_atk = 10
        self._board.cost_def = 10
        for t in range(config.tower_types):
            self._board.cost_def += config.tower_cost[t][0]
            succ = False
            for i in range(self._board.map_size):
                for j in range(self._board.map_size):
                    if self._board.map[6, i, j] == 0:
                        self._board.tower_build(t, [i, j])
                        succ = True
                        break
                if succ:
                    break
                        
        self._board.step()
    
    def random_enemy_lv0(self):
        if self.random_agent:
            import random
            cluster = [random.randint(0, config.enemy_types) for _ in range(hyper_parameters.max_cluster_length)]
            road = random.randint(0, self.num_roads-1)
        else:
            cluster = self.np_random.randint(0, config.enemy_types, [hyper_parameters.max_cluster_length], dtype=np.int64)
            road = self.np_random.randint(self.num_roads)
        self._board.summon_cluster(
            cluster, road
        )
    
    def random_enemy_lv1(self):
        if self.random_agent:
            import random
            t = random.randint(0, config.enemy_types-1)
            road = random.randint(0, self.num_roads-1)
        else:
            t = self.np_random.randint(0, config.enemy_types)
            road = self.np_random.randint(self.num_roads)
        cluster = np.full([hyper_parameters.max_cluster_length], t)
        self._board.summon_cluster(
            cluster, road
        )

                
    def random_tower_lv0(self):
        if self.random_agent:
            import random
            r = random.randint(0, self.map_size-1)
            c = random.randint(0, self.map_size-1)
            t = random.randint(0, config.tower_types-1)
        else:
            r, c = self.np_random.randint(0, self.map_size, [2,])
            t = self.np_random.randint(0, config.tower_types)
        self._board.tower_build(t, [r,c])

    def random_tower_lv1(self):
        dp = [[r, c] for r in range(-2, 3) for c in range(-2, 3)]
        if getattr(self, '__wait_for_cost_rt1', None) is not None:
            if self._board.tower_build(*self.__wait_for_cost_rt1):
                self.__wait_for_cost_rt1 = None
            else:
                if self._board.fail_code != FC.COST_SHORTAGE:
                    self.__wait_for_cost_rt1 = None
            return

        if self.random_agent:
            import random
            act = random.randint(0, 2)
        else:
            act = self.np_random.randint(0, 3)

        if act == 0:
            roads = []
            for r in range(self.map_size):
                for c in range(self.map_size):
                    if self._board.map[0, r, c] == 1:
                        roads.append([r, c])
            
            if self.random_agent:
                random.shuffle(roads)
                t = random.randint(0, config.tower_types-1)
            else:
                self.np_random.shuffle(roads)
                t = self.np_random.randint(0, config.tower_types)

            for r, c in roads:
                if self.random_agent:
                    d = dp[random.randint(0, len(dp)-1)]
                else:
                    d = dp[self.np_random.randint(0, len(dp))]
                pos = [r+d[0], c+d[1]]
                if not self._board.is_valid_pos(pos):
                    continue
                if self._board.tower_build(t, pos):
                    return
                else:
                    if self._board.fail_code == FC.COST_SHORTAGE:
                        self.__wait_for_cost_rt1 = [t, pos]
                        return
        elif act == 1:
            if len(self._board.towers) == 0:
                return
            if self.random_agent:
                id = random.randint(0, len(self._board.towers)-1)
            else:
                id = self.np_random.randint(0, len(self._board.towers))
            self._board.tower_lvup(self._board.towers[id].loc)
        elif act == 2:
            if len(self._board.towers) == 0:
                return
            if self.random_agent:
                if random.random() > .01:
                    return
                id = random.randint(0, len(self._board.towers)-1)
            else:
                if self.np_random.random() > .01:
                    return
                id = random.randint(0, len(self._board.towers)-1)
            self._board.tower_destruct(self._board.towers[id].loc)
        else:
            return

    def random_tower_lv2(self):
        dp = [[r, c] for r in range(-2, 3) for c in range(-2, 3)]
        if getattr(self, '__wait_for_cost_rt2', None) is not None:
            if self._board.tower_build(*self.__wait_for_cost_rt2):
                self.__wait_for_cost_rt2 = None
            else:
                if self._board.fail_code != FC.COST_SHORTAGE:
                    self.__wait_for_cost_rt2 = None
            return

        if self.random_agent:
            import random
            act = random.randint(0, 2)
        else:
            act = self.np_random.randint(0, 3)

        if act == 0:
            roads = []
            et = list(map(lambda x: x.type, self._board.enemies))
            if len(et) == 0:
                return
            types, nums = np.unique(et, return_counts=True)
            ratio = nums.astype(np.float32) / np.sum(nums)
            if self.random_agent:
                p = random.random()
            else:
                p = self.np_random.random()
            for i in range(4):
                if p < ratio[i]:
                    t = types[i]
                    break
                else:
                    p -= ratio[i]
            
            t = [2, 0, 1, 0][t]
            if self.random_agent:
                p = random.random()
            else:
                p = self.np_random.random()
            if p < 0.2:
                t = 3

            for r in range(self.map_size):
                for c in range(self.map_size):
                    if self._board.map[0, r, c] == 1:
                        roads.append([r, c])
            
            if self.random_agent:
                random.shuffle(roads)
            else:
                self.np_random.shuffle(roads)

            for r, c in roads:
                if self.random_agent:
                    d = dp[random.randint(0, len(dp)-1)]
                else:
                    d = dp[self.np_random.randint(0, len(dp))]
                pos = [r+d[0], c+d[1]]
                if not self._board.is_valid_pos(pos):
                    continue
                if self._board.tower_build(t, pos):
                    return
                else:
                    if self._board.fail_code == FC.COST_SHORTAGE:
                        self.__wait_for_cost_rt2 = [t, pos]
                        return
        elif act == 1:
            if len(self._board.towers) == 0:
                return
            if self.random_agent:
                id = random.randint(0, len(self._board.towers)-1)
            else:
                id = self.np_random.randint(0, len(self._board.towers))
            if self._board.tower_lvup(self._board.towers[id].loc):
                return
        elif act == 2:
            if len(self._board.towers) == 0:
                return
            if self.random_agent:
                if random.random() > .01:
                    return
                id = random.randint(0, len(self._board.towers)-1)
            else:
                if self.np_random.random() > .01:
                    return
                id = random.randint(0, len(self._board.towers)-1)
            if self._board.tower_destruct(self._board.towers[id].loc):
                return
        else:
            return


    def render(self, mode="human"):
        return self._board.render(mode)

