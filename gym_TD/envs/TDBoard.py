import numpy as np

from gym_TD.utils import logger
from gym_TD.utils import fail_code as FC
from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.envs.TDElements import *

from gym_TD.envs import TDRoadGen

class TDBoard(object):
    '''
    The implementation of basic rule and states of TD game.
    '''
    def __init__(self, map_size, num_roads, np_random):
        '''
        Create the game field
        map_size: The length of edge of the field.
        num_roads: The number of roads generated. num_roads must be one of {1, 2, 3}
        np_random: A Numpy BitGenerator or None. Used to control the field generation.
        '''
        self.map_size = map_size

        if np_random is None:
            from gym.utils import seeding
            np_random, _ = seeding.np_random(None)

        self.map = np.zeros(shape=(7, map_size, map_size), dtype=np.int32)
        # is road, is road1, is road2, is road3, dist to end, where to go, how many towers nearby
        roads = TDRoadGen.create_road(np_random, map_size, num_roads)
        self.num_roads = num_roads
        
        self.start = [r[0] for r in roads]
        self.end = roads[0][-1]

        for i, road in enumerate(roads):
            last = None
            for p in road: # from start to end
                self.map[0, p[0], p[1]] = 1
                self.map[i+1, p[0], p[1]] = 1
                self.map[6, p[0], p[1]] = 1
                if last is not None:
                    if p[0] - last[0] == 0:
                        if p[1] - last[1] == 1:
                            direct = 0
                        else:
                            direct = 1
                    elif p[0] - last[0] == 1:
                        direct = 2
                    else:
                        direct = 3
                    self.map[5, last[0], last[1]] = direct
                last = p
            dist = 0
            for p in reversed(road): # from end to start
                self.map[4, p[0], p[1]] = dist
                dist += 1
        
        logger.debug('Board', 'Map: {}', self.map)

        self.enemy_LP = np.zeros((4, config.enemy_types, self.map_size, self.map_size), dtype=np.float32)
        # lowest, highest, average LP, count of enemies

        self.enemies = []
        self.towers = []
        self.cost_def = config.defender_init_cost
        self.cost_atk = config.attacker_init_cost
        self.max_cost = config.max_cost
        self.base_LP = config.base_LP
        self.max_base_LP = config.base_LP

        self.viewer = None

        self.steps = 0
        self.progress = 0.

        self.__fail_code = FC.SUCCESS
    
    @property
    def fail_code(self):
        return self.__fail_code
    
    def get_states(self):
        '''
        Channels:
        0: is road
        1: is road 1
        2: is road 2
        3: is road 3
        4: is end point
        5: LP ratio of end point
        6: is start point 1
        7: is start point 2
        8: is start point 3
        9: distance to end point
        10: is tower
        11: defender cost
        12: attacker cost
        13: progress of the game
        14: could build tower
        [15, 15+# tower lv): tower level is [0, # tower lv)
        [15+# tower lv, 15+# tower lv+# tower type): tower type is [0, # tower type)
        [15+# tower lv+# tower type, 15+# tower lv+2 # tower type): tower of type could be built
        [a, a+# enemy type): lowest enemy LP of type [0, # enemy type)
        [a+# enemy type, a+2 # enemy type): highest enemy LP of type [0, # enemy type)
        [a+2 # enemy type, a+3 # enemy type): average enemy LP of type [0, # enemy type)
        [a+3 # enemy type, a+4 # enemy type): number of enemies of type [0, # enemy type)
        [a+4 # enemy type, a+5 # enemy type): how many enemies could be summoned of type [0, # enemy type)
        '''
        s = np.zeros(shape=self.state_shape, dtype=np.float32)
        s[0:4] = self.map[0:4]
        s[4, self.end[0], self.end[1]] = 1
        if self.max_base_LP is None:
            s[5] = 1.
        else:
            s[5] = self.base_LP / self.max_base_LP
        for i, start in enumerate(self.start):
            s[6+i, start[0], start[1]] = 1
        s[9] = self.map[4]
        s[9] /= (np.max(self.map[4]) + 1)
        s[11] = self.cost_def / self.max_cost
        s[12] = self.cost_atk / self.max_cost
        s[13] = self.progress
        if len(self.towers) < config.max_num_tower:
            s[14] = (self.map[6] == 0)
        
        tower_lv_base = 15
        tower_type_base = tower_lv_base + config.max_tower_lv + 1
        tower_build_base = tower_type_base + config.tower_types
        for t in self.towers:
            s[tower_lv_base+t.lv, t.loc[0], t.loc[1]] = 1
            s[tower_type_base+t.type, t.loc[0], t.loc[1]] = 1
        for t in range(config.tower_types):
            s[tower_build_base + t] = 1 if self.cost_def >= config.tower_cost[t][0] else 0
        
        enemy_channel_base = tower_build_base + config.tower_types
        can_summon_base = enemy_channel_base + 4 * config.enemy_types

        s[enemy_channel_base: can_summon_base] = self.enemy_LP.reshape((4*config.enemy_types, self.map_size, self.map_size))
        for t in range(config.enemy_types):
            s[can_summon_base + t] = self.cost_def / config.enemy_cost[t][0] / hyper_parameters.max_cluster_length

        return s
    
    def get_atk_valid_mask(self):
        '''
        Return the valid mask of actions of attacker
        Shape = (max_num_of_roads, config.enemy_types,)
        DType = np.bool
        mask[i,j] == True for valid action i,j and == False for invalid action i,j
        action i,j is to summon the enemy of type j at road i
        '''
        m = np.zeros((hyper_parameters.max_num_of_roads, config.enemy_types,), dtype = np.bool)
        lv = 1 if self.progress >= config.enemy_upgrade_at else 0
        for i, c in enumerate(config.enemy_cost):
            if c[lv] < self.cost_atk:
                m[:self.num_roads, i] = True
        return m

    def get_def_valid_mask(self):
        '''
        Return the valid mask of actions of defender
        Shape = (config.tower_types + 2, map_size, map_size)
        DType = np.bool
        mask[i, j, k]:
            (j, k): coordinate
            i \in [0, config.tower_types): can build tower of type i
            i == config.tower_types: can upgrade the tower
            i == config.tower_types+1: can destruct the tower
        True for valid action and False for invalid action
        '''
        m = np.zeros((config.tower_types + 2, self.map_size, self.map_size), dtype=np.bool)
        for t in range(config.tower_types):
            c = config.tower_cost[t][0]
            if c > self.cost_def:
                continue
            for i in range(self.map_size):
                for j in range(self.map_size):
                    if self.map[6, i, j] == 0:
                        m[t, i, j] = True
        for t in self.towers:
            if t.lv < config.max_tower_lv and \
                config.tower_cost[t.type][t.lv+1] <= self.cost_def:
                m[config.tower_types, t.loc[0], t.loc[1]] = True
            m[config.tower_types+1, t.loc[0], t.loc[1]] = True
        return m
    
    @staticmethod
    def n_channels():
        '''
        Return how many channels there will be in the state tensor

        >>> TDBoard.n_channels()
        45
        '''
        return 15 + 2 * config.tower_types + config.max_tower_lv + 1 + 5 * config.enemy_types

    @property
    def state_shape(self):
        '''
        Return the shape of state tensor

        >>> board = TDBoard(10, 2, None, 10, 10, 100, 5)
        >>> board.state_shape
        (45, 10, 10)
        '''
        return (self.n_channels(), self.map_size, self.map_size)
    
    def is_valid_pos(self, pos):
        '''
        Check if the position is valid for this field

        >>> board = TDBoard(10, 2, None, 10, 10, 100, 5)
        >>> board.is_valid_pos([10, 2])
        False
        >>> board.is_valid_pos([-1, 3])
        False
        >>> board.is_valid_pos([5, 10])
        False
        >>> board.is_valid_pos([4, -1])
        False
        >>> board.is_valid_pos([2, 3])
        True
        '''
        return pos[0] >= 0 and pos[0] < self.map_size and pos[1] >= 0 and pos[1] < self.map_size
    
    def summon_enemy(self, t, start_id):
        start = self.start[start_id]
        lv = 1 if self.progress >= config.enemy_upgrade_at else 0
        logger.debug('Board', 'summon_enemy: {}, {} ({},{})', t, start_id, start[0], start[1])
        e = create_enemy(t, start, self.map[4, start[0], start[1]], lv)
        if self.cost_atk < e.cost:
            logger.verbose('Board', 'Summon enemy {} failed due to cost shortage', t)
            self.__fail_code = FC.COST_SHORTAGE
            return False
        logger.debug('Board', 'Summon enemy {}', t)
        self.enemies.append(e)
        self.cost_atk -= e.cost
        self.__fail_code = FC.SUCCESS
        return True
    
    def summon_cluster(self, types, start_id):
        start = self.start[start_id]
        lv = 1 if self.progress >= config.enemy_upgrade_at else 0
        logger.debug('Board', 'summon_cluster: {}, {} ({},{})', types, start_id, start[0], start[1])
        tried = False
        summoned = False
        real_act = []
        for t in types:
            if t == config.enemy_types:
                real_act.append(t)
                continue
            tried = True
            e = create_enemy(t, start, self.map[4, start[0], start[1]], lv)
            if self.cost_atk < e.cost:
                real_act.append(config.enemy_types)
            else:
                self.cost_atk -= e.cost
                self.enemies.append(e)
                summoned = True
                real_act.append(t)
        if (not summoned) and tried:
            logger.verbose('Board', 'Summon cluster {} failed due to cost shortage', types)
            self.__fail_code = FC.COST_SHORTAGE
            return False, real_act
        self.__fail_code = FC.SUCCESS
        return True, real_act

    def tower_build(self, t, loc):
        p = create_tower(t, loc)
        if self.cost_def < p.cost:
            logger.verbose('Board', 'Build tower {} ({},{}) failed due to cost shortage', t, loc[0], loc[1])
            self.__fail_code = FC.COST_SHORTAGE
            return False
        if len(self.towers) >= config.max_num_tower:
            logger.verbose('Board', 'Cannot build tower because reached upper limit')
            self.__fail_code = FC.TOWER_NUMBER_LIMIT
            return False
        if self.map[6, loc[0], loc[1]] > 0:
            logger.verbose('Board', 'Cannot build tower {} at ({},{})', t, loc[0], loc[1])
            self.__fail_code = FC.INVALID_POSITION
            return False
        logger.debug('Board', 'Build tower {} ({},{})', t, loc[0], loc[1])
        self.towers.append(p)
        self.cost_def -= p.cost
        for i in range(-config.tower_distance, config.tower_distance+1):
            for j in range(-config.tower_distance, config.tower_distance+1):
                if abs(i) + abs(j) <= config.tower_distance:
                    r, c = loc[0]+i, loc[1]+j
                    if r < 0 or r >= self.map_size or c < 0 or c >= self.map_size:
                        continue
                    self.map[6, r, c] += 1
        self.__fail_code = FC.SUCCESS
        return True

    def tower_lvup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.lv >= config.max_tower_lv:
                    logger.verbose('Board', 'Tower LV up ({},{}) failed because lv max', loc[0], loc[1])
                    self.__fail_code = FC.LV_MAX
                    return False
                cost = config.tower_cost[t.type][t.lv+1]
                if self.cost_def < cost:
                    logger.verbose('Board', 'Tower LV up ({},{}) failed due to cost shortage', loc[0], loc[1])
                    self.__fail_code = FC.COST_SHORTAGE
                    return False
                if not upgrade_tower(t):
                    logger.verbose('Board', 'Tower LV up ({},{}) failed', loc[0], loc[1])
                    self.__fail_code = FC.LV_MAX
                    return False
                logger.debug('Board', 'Tower LV up ({},{})', loc[0], loc[1])
                self.cost_def -= cost
                self.__fail_code = FC.SUCCESS
                return True
        logger.verbose('Board', 'No tower at ({},{})', loc[0], loc[1])
        self.__fail_code = FC.UNKNOWN_TARGET
        return False

    def tower_destruct(self, loc):
        for t in self.towers:
            if loc == t.loc:
                self.cost_def += t.cost * config.tower_destruct_return
                self.cost_def = min(self.cost_def, self.max_cost)
                self.towers.remove(t)
                logger.debug('Board', 'Destruct tower ({},{})', loc[0], loc[1])

                for i in range(-config.tower_distance, config.tower_distance+1):
                    for j in range(-config.tower_distance, config.tower_distance+1):
                        if abs(i) + abs(j) <= config.tower_distance:
                            r, c = loc[0]+i, loc[1]+j
                            if r < 0 or r >= self.map_size or c < 0 or c >= self.map_size:
                                continue
                            self.map[6, r, c] -= 1

                self.__fail_code = FC.SUCCESS
                return True
        logger.verbose('Board', 'No tower at ({},{})', loc[0], loc[1])
        self.__fail_code = FC.UNKNOWN_TARGET
        return False
    
    def step(self):
        # Forward the game one time step.
        # Return the reward for the defender.
        reward = 0.
        reward += config.reward_time
        self.steps += 1
        self.progress = self.steps / hyper_parameters.max_episode_steps
        logger.debug('Board', 'Step: {}->{}', self.steps-1, self.steps)

        to_del = []
        self.enemies.sort(key=lambda x: x.dist - x.margin)
        for t in self.towers:
            t.cd -= 1 # cool down
            if t.cd > 0:
                continue
            killed = t.attack(self.enemies)
            if t.cd < 0:
                t.cd = 0
            to_del += [e for e in killed if e not in to_del]
        
        reward += config.reward_kill * len(to_del)
        for e in to_del:
            self.enemies.remove(e)

        dp = [[0,1], [0,-1], [1,0], [-1,0]]
        toremove = []
        for e in self.enemies:
            if e.slowdown > 0:
                e.margin += e.speed * config.frozen_ratio
                e.slowdown -= 1
            else:
                e.margin += e.speed
            while e.margin >= 1.:
                # move to next grid
                e.margin -= 1.
                d = self.map[5, e.loc[0], e.loc[1]]
                p = [
                    e.loc[0] + dp[d][0],
                    e.loc[1] + dp[d][1]
                ]
                e.move(p, self.map[4, p[0], p[1]])
                if e.loc == self.end:
                    if self.base_LP is not None and self.base_LP > 0:
                        reward -= config.penalty_leak
                    logger.debug('Board', 'Leak {}', e.type)
                    toremove.append(e)
                    if self.base_LP is not None:
                        logger.debug('Board', 'LP: {} -> {}', self.base_LP, max(self.base_LP-1, 0))
                        self.base_LP = max(self.base_LP-1, 0)
                    break
        for e in toremove:
            self.enemies.remove(e)
        
        if self.progress >= 0.5:
            attacker_cost_rate = config.attacker_cost_final_rate
        else:
            attacker_cost_rate = config.attacker_cost_init_rate * (1. - self.progress) + config.attacker_cost_final_rate * self.progress
        self.cost_atk = min(self.cost_atk+attacker_cost_rate, self.max_cost)
        self.cost_def = min(self.cost_def+config.defender_cost_rate, self.max_cost)

        self.enemy_LP[:] = 0
        self.enemy_LP[0] = 1.
        for e in self.enemies:
            r = e.LP / e.maxLP
            self.enemy_LP[0, e.type, e.loc[0], e.loc[1]] = min(self.enemy_LP[0, e.type, e.loc[0], e.loc[1]], r)
            self.enemy_LP[1, e.type, e.loc[0], e.loc[1]] = max(self.enemy_LP[1, e.type, e.loc[0], e.loc[1]], r)
            self.enemy_LP[2, e.type, e.loc[0], e.loc[1]] += r
            self.enemy_LP[3, e.type, e.loc[0], e.loc[1]] += 1
        self.enemy_LP[0] = np.where(self.enemy_LP[3] > 0, self.enemy_LP[0], np.zeros_like(self.enemy_LP[0]))
        self.enemy_LP[2] = np.where(self.enemy_LP[3] > 0, self.enemy_LP[2] / self.enemy_LP[3], np.zeros_like(self.enemy_LP[2]))
        self.enemy_LP[3] /= hyper_parameters.max_cluster_length

        logger.debug('Board', 'Reward: {}', reward)
        return reward
    
    def done(self):
        '''
        Check if the game has finished

        >>> board = TDBoard(10, 2, None, 10, 10, 100, 5)
        >>> board.done()
        False
        >>> board.base_LP = 0
        >>> board.done()
        True
        >>> board.base_LP, board.steps = 5, 1200
        >>> board.done()
        True
        '''
        return (self.base_LP is not None and self.base_LP <= 0) \
            or self.steps >= hyper_parameters.max_episode_steps
    
    def render(self, mode):
        screen_width = 800
        screen_height = 800
        cost_bar_height = 15
        sample_height = 15
        enemy_colors = [
            [1, 0, .5],
            [1, .5, 1],
            [.5, 0, 1],
            [.71, .29, .71]
        ]
        bar_colors = [
            [0,.5,1], # min LP
            [1,.5,0], # max LP
            [1,1,.5], # avg LP
            [.5,1,1] # nums
        ]   
        tower_colors = [
            [.5, 1, 0],
            [0, .5, 0],
            [0, 1, .5],
            [.29, .71, .29]
        ]
        if self.viewer is None: #draw permanent elements here
            from gym.envs.classic_control import rendering
            
            self.viewer = rendering.Viewer(screen_width, screen_height+cost_bar_height+sample_height)
            # create viewer
            # background = rendering.FilledPolygon([
            #     (0, 0), (0, screen_height),
            #     (screen_width, screen_height), (screen_width, 0)
            # ])
            # background.set_color(1,1,1)
            # self.viewer.add_geom(background)
            # # draw background

            for i in range(len(enemy_colors)):
                left, right, top, bottom = \
                    sample_height*i,\
                    sample_height*(i+1),\
                    screen_height+sample_height,\
                    screen_height
                enemy_sample = rendering.FilledPolygon([
                            (left, bottom), (left, top),
                            (right, top), (right, bottom)
                        ])
                enemy_sample.set_color(*enemy_colors[i])
                self.viewer.add_geom(enemy_sample)
            left, right = \
                sample_height * len(enemy_colors),\
                sample_height * (len(enemy_colors) + 1)
            for i in range(len(bar_colors)):
                top = screen_height + sample_height * (i+1) / (len(bar_colors)+1)
                bar_sample = rendering.Line(
                    (left, top), (right, top)
                )
                bar_sample.set_color(*bar_colors[i])
                self.viewer.add_geom(bar_sample)
            for i in range(len(tower_colors)):
                left, right, top, bottom = \
                    sample_height*(i+len(enemy_colors)+1),\
                    sample_height*(i+len(enemy_colors)+2),\
                    screen_height+sample_height,\
                    screen_height
                tower_sample = rendering.FilledPolygon([
                            (left, bottom), (left, top),
                            (right, top), (right, bottom)
                        ])
                tower_sample.set_color(*tower_colors[i])
                self.viewer.add_geom(tower_sample)

            for i in range(self.map_size+1):
                gridLine = rendering.Line(
                    (0, screen_height * i // self.map_size),
                    (screen_width, screen_height * i // self.map_size)
                )
                gridLine.set_color(0, 0, 0) #black
                self.viewer.add_geom(gridLine)
            for i in range(self.map_size+1):
                gridLine = rendering.Line(
                    (screen_width * i // self.map_size, 0),
                    (screen_width * i // self.map_size, screen_height)
                )
                gridLine.set_color(0, 0, 0) #black
                self.viewer.add_geom(gridLine)
            # draw grid lines
            for r in range(self.map_size):
                for c in range(self.map_size):
                    if self.map[0,r,c] == 1:
                        left, right, top, bottom = \
                            screen_width * c // self.map_size, \
                            screen_width * (c+1) // self.map_size, \
                            screen_height * (r+1) // self.map_size, \
                            screen_height * r // self.map_size
                        road = rendering.FilledPolygon([
                            (left, bottom), (left, top),
                            (right, top), (right, bottom)
                        ])
                        road.set_color(.2, .2, .2) #gray
                        self.viewer.add_geom(road)
            # draw road
            
            for start in self.start:
                left, right, top, bottom = \
                    screen_width * start[1] // self.map_size, \
                    screen_width * (start[1]+1) // self.map_size, \
                    screen_height * (start[0]+1) // self.map_size, \
                    screen_height * start[0] // self.map_size
                startpoint = rendering.FilledPolygon([
                    (left, bottom), (left, top),
                    (right, top), (right, bottom)
                ])
                startpoint.set_color(1, 0, 0) #red
                self.viewer.add_geom(startpoint)
            # draw startpoint
            left, right, top, bottom = \
                screen_width * self.end[1] // self.map_size, \
                screen_width * (self.end[1]+1) // self.map_size, \
                screen_height * (self.end[0]+1) // self.map_size, \
                screen_height * self.end[0] // self.map_size
            endpoint = rendering.FilledPolygon([
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ])
            endpoint.set_color(0, 0, 1) #blue
            self.viewer.add_geom(endpoint)
            # draw endpoint
        
        # draw changable elements here
        # attacker cost bar
        left, right, top, bottom = \
            0, screen_width*(self.cost_atk/self.max_cost)/3, \
            screen_height+cost_bar_height+sample_height, screen_height+sample_height
        self.viewer.draw_polygon(
            v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ],
            color=(1, .3, 0)
        )
        # defender cost bar
        left, right, top, bottom = \
            screen_width/3, screen_width*(1+self.cost_def/self.max_cost)/3, \
            screen_height+cost_bar_height+sample_height, screen_height+sample_height
        self.viewer.draw_polygon(
            v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ],
            color=(0, .3, 1)
        )
        # progress bar
        left, right, top, bottom = \
            screen_width*2/3, screen_width*(2+self.progress)/3, \
            screen_height+cost_bar_height+sample_height, screen_height+sample_height
        self.viewer.draw_polygon(
            v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ],
            color=(1, 1, 0)
        )

        
        for r in range(self.map_size):
            for c in range(self.map_size):
                if self.map[6,r,c] >= 1 and self.map[0,r,c] == 0:
                    left, right, top, bottom = \
                        screen_width * c // self.map_size, \
                        screen_width * (c+1) // self.map_size, \
                        screen_height * (r+1) // self.map_size, \
                        screen_height * r // self.map_size
                    self.viewer.draw_polygon(
                        v=[
                            (left, bottom), (left, top),
                            (right, top), (right, bottom)
                        ],
                        color = (.8, .8, .8)
                    )
        # show where tower could not be built

        block_width = screen_width // self.map_size
        block_height = screen_height // self.map_size
        
        enemy_offset = [
            [0, 0], # left bottom
            [block_width//2, 0], # right bottom
            [0, block_height//2], # left up
            [block_width//2, block_height//2] # right up
        ]
        mg16 = block_width / 16 # margin 1-16th
        mg8 = block_width / 8
        mg4 = block_width / 4
        mg316 = mg16*3
        mg716 = mg16*7
        mg38 = mg8*3
        bar_length = mg4

        for r in range(self.map_size):
            for c in range(self.map_size):
                if self.map[0,r,c] == 1:
                    left, bottom = \
                        screen_width * c // self.map_size, \
                        screen_height * r // self.map_size
                    for t in range(config.enemy_types):
                        if self.enemy_LP[1, t, r, c] == 0:
                            continue
                        l, b = left + enemy_offset[t][0], bottom + enemy_offset[t][1]
                        self.viewer.draw_polygon(
                            v=[
                                (l+mg8, b+mg316), (l+mg8, b+mg716),
                                (l+mg38, b+mg716), (l+mg38, b+mg316)
                            ],
                            color=enemy_colors[t]
                        )
                        self.viewer.draw_line(
                            (l+mg8, b+mg16), (l+mg8+bar_length*self.enemy_LP[3,t,r,c], b+mg16),
                            color=bar_colors[3]
                        )
                        self.viewer.draw_line(
                            (l+mg8, b+mg8), (l+mg8+bar_length*self.enemy_LP[1,t,r,c], b+mg8),
                            color=bar_colors[1]
                        )
                        self.viewer.draw_line(
                            (l+mg8, b+mg8), (l+mg8+bar_length*self.enemy_LP[2,t,r,c], b+mg8),
                            color=bar_colors[2]
                        )
                        self.viewer.draw_line(
                            (l+mg8, b+mg8), (l+mg8+bar_length*self.enemy_LP[0,t,r,c], b+mg8),
                            color=bar_colors[0]
                        )
        # draw enemies
        
        for t in self.towers:
            left, right, top, bottom = \
                screen_width * t.loc[1] // self.map_size, \
                screen_width * (t.loc[1]+1) // self.map_size, \
                screen_height * (t.loc[0]+1) // self.map_size, \
                screen_height * t.loc[0] // self.map_size
            self.viewer.draw_polygon(
                v=[
                    (left, bottom), (left, top),
                    (right, top), (right, bottom)
                ],
                color = tower_colors[t.type]
            )
            maxlv = config.max_tower_lv
            self.viewer.draw_polygon(
                v=[
                    (left+block_width//3, bottom),
                    (left+block_width//3, bottom+block_height//(maxlv+1)*(t.lv+1)),
                    (left+block_width*2//3, bottom+block_height//(maxlv+1)*(t.lv+1)),
                    (left+block_width*2//3, bottom),
                ],
                color = (0, 1, 0)
            )
            if t.cd > 0:
                self.viewer.draw_line(
                    (left, bottom), (left+block_width*t.cd/t.intv, bottom), color=(0,1,0)
                )
        # draw towers
            
        if self.base_LP is not None:
            left, bottom = \
                screen_width * self.end[1] // self.map_size, \
                screen_height * self.end[0] // self.map_size
            top = bottom + screen_height // self.map_size // 10
            right = left + int(screen_width // self.map_size * self.base_LP / self.max_base_LP + .5)
            self.viewer.draw_polygon(
                v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
                ],
                color = (0, 0.8, 0.8)
            )
            # draw LP bar of base point

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        '''
        Close the render viewer
        '''
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    rng = np.random.RandomState()
    rng.seed(1024)
    board = TDBoard(
        10, 2, rng,
        config.defender_init_cost,
        config.attacker_init_cost,
        config.max_cost,
        config.base_LP
    )

    import doctest
    doctest.testmod(verbose=True)

    state = board.get_states()
    
    ground_truth = np.zeros(board.state_shape, dtype=np.float32)
    road0 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    road0 = np.asarray(road0, dtype=np.float32)
    road1 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ]
    road1 = np.asarray(road1, dtype=np.float32)
    dist = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 11, 12, 13],
        [0, 2, 0, 0, 0, 0, 0, 10, 0, 0],
        [0, 3, 4, 5, 6, 7, 8, 9, 0, 0],
        [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 0, 0, 0, 0, 0]
    ]
    dist = np.asarray(dist, dtype=np.float32)
    road = road0 + road1
    road = np.where(road > 0, np.ones_like(road), np.zeros_like(road))
    ground_truth[0] = road
    ground_truth[1] = road0
    ground_truth[2] = road1
    ground_truth[4, 4, 0] = 1    
    ground_truth[6, 4, 9] = 1
    ground_truth[7, 9, 4] = 1
    ground_truth[5] = 1
    ground_truth[9] = dist / 14
    ground_truth[11] = config.defender_init_cost / config.max_cost
    ground_truth[12] = config.attacker_init_cost / config.max_cost
    ground_truth[13] = 0
    ground_truth[14] = 1 - road
    ground_truth[21] = 1
    for i in range(4):
        ground_truth[41+i] = config.defender_init_cost / config.enemy_cost[i][0] / hyper_parameters.max_cluster_length
    
    assert np.all(state == ground_truth)
    for i in range(4):
        for j in range(2):
            assert board.summon_enemy(i, j) == False
    

    # board.render('human')
    # input()
    print('passed')
