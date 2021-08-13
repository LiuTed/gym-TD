import numpy as np

from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym_TD.utils import logger
from gym_TD.envs.TDParam import config, hyper_parameters

from gym_TD.envs import TDRoadGen

class Enemy(object):
    def __init__(self, maxLP, speed, cost, loc, t=None):
        self.maxLP = self.LP = maxLP
        self.speed = speed
        self.cost = cost
        self.loc = loc
        self.margin = 0.
        self.type = t
    
    def setLoc(self, newLoc):
        self.loc = newLoc
    def damage(self, dmg):
        self.LP -= dmg
        if self.LP <= 0:
            self.LP = 0
    @property
    def alive(self):
        return self.LP > 0

class EnemyBalance(Enemy):
    def __init__(self, loc):
        super(EnemyBalance, self).__init__(
            config.enemy_balance_LP,
            config.enemy_balance_speed,
            config.enemy_balance_cost,
            loc,
            0
        )
class EnemyStrong(Enemy):
    def __init__(self, loc):
        super(EnemyStrong, self).__init__(
            config.enemy_strong_LP,
            config.enemy_strong_speed,
            config.enemy_strong_cost,
            loc,
            1
        )
class EnemyFast(Enemy):
    def __init__(self, loc):
        super(EnemyFast, self).__init__(
            config.enemy_fast_LP,
            config.enemy_fast_speed,
            config.enemy_fast_cost,
            loc,
            2
        )
def _create_enemy(t, loc):
    classes = {
        0: EnemyBalance,
        1: EnemyStrong,
        2: EnemyFast
    }
    return classes[t](loc)

class Tower(object):
    def __init__(self, atk, rge, loc, cost):
        self.atklv = 0
        self.atk = atk
        self.rangelv = 0
        self.rge = rge
        self.loc = loc
        self.cost = cost
        self.cd = 0

    def atkUp(self, atk, cost):
        self.atk += atk
        self.atklv += 1
        self.cost += cost
    def rangeUp(self, rge, cost):
        self.rge += rge
        self.rangelv += 1
        self.cost += cost

def _create_tower(loc):
    return Tower(
        config.tower_basic_ATK,
        config.tower_basic_range,
        loc,
        config.tower_basic_cost
    )

class TDBoard(object):
    '''
    The implementation of basic rule and states of TD game.
    '''
    def __init__(self, map_size, num_roads, np_random, cost_def, cost_atk, max_cost, base_LP):
        self.map_size = map_size

        self.map = np.zeros(shape=(map_size, map_size, 6), dtype=np.int32)
        # is road, is road1, is road2, is road3, dist to end, where to go
        roads = TDRoadGen.create_road(np_random, map_size, num_roads)
        
        self.start = [r[0] for r in roads]
        self.end = roads[0][-1]

        for i, road in enumerate(roads):
            last = None
            for p in road: # from start to end
                self.map[p[0], p[1], 0] = 1
                self.map[p[0], p[1], i+1] = 1
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
                    self.map[last[0], last[1], 5] = direct
                last = p
            dist = 0
            for p in reversed(road): # from end to start
                self.map[p[0], p[1], 4] = dist
                dist += 1
        
        logger.debug('B', 'Map: {}', self.map)

        self.cnt = np.zeros(shape=(self.map_size, self.map_size, 3), dtype=np.int32)
        # how many enemies of type [2] at location ([0], [1])

        self.enemies = []
        self.towers = []
        self.cost_def = cost_def
        self.cost_atk = cost_atk
        self.max_cost = max_cost
        self.base_LP = base_LP
        self.max_base_LP = base_LP

        self.viewer = None

        self.steps = 0
    
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
        [13, 13+natk): tower atk lv is [0, natk]
        [13+natk, 13+natk+nrange): tower range lv is [0, nrange]
        [a, a+ol): LP ratio of enemies of type 0
        [a+ol, a+2ol): LP ratio of enemies of type 1
        [a+2ol, a+3ol): LP ratio of enemies of type 2
        [a+3ol, a+4ol): distance margin of enemies of type 0
        [a+4ol, a+5ol): distance margin of enemies of type 1
        [a+5ol, a+6ol): distance margin of enemies of type 2
        '''
        ptr = np.zeros(shape=(self.map_size, self.map_size, 3), dtype=np.int32)
        s = np.zeros(shape=self.state_shape, dtype=np.float32)
        s[:,:,0:4] = self.map[:,:,0:4]
        s[self.end[0], self.end[1], 4] = 1
        s[:,:,5] = self.base_LP / self.max_base_LP
        for i, start in enumerate(self.start):
            s[start[0], start[1], 6+i] = 1
        s[:,:,9] = self.map[:,:,4] / np.max(self.map[:,:,4])
        s[:,:,11] = self.cost_def/self.max_cost
        s[:,:,12] = self.cost_atk/self.max_cost
        
        tower_atk_base = 13
        tower_range_base = tower_atk_base + len(config.tower_ATKUp_list) + 1
        for t in self.towers:
            s[t.loc[0], t.loc[1], 10] = 1
            s[t.loc[0], t.loc[1], tower_atk_base+t.atklv] = 1
            s[t.loc[0], t.loc[1], tower_range_base+t.rangelv] = 1

        enemy_base = tower_range_base + len(config.tower_rangeUp_list) + 1
        for e in self.enemies:
            n = ptr[e.loc[0], e.loc[1], e.type]
            s[e.loc[0], e.loc[1], enemy_base+hyper_parameters.enemy_overlay_limit*e.type+n] = e.LP/e.maxLP
            s[e.loc[0], e.loc[1], enemy_base+hyper_parameters.enemy_overlay_limit*(e.type+3)+n] = e.margin
            ptr[e.loc[0], e.loc[1], e.type] += 1
        return s
    
    @staticmethod
    def n_channels():
        return 15 + len(config.tower_ATKUp_list) + len(config.tower_rangeUp_list) + 6*hyper_parameters.enemy_overlay_limit
    @property
    def state_shape(self):
        return (self.map_size, self.map_size, self.n_channels())
    
    def is_valid_pos(self, pos):
        return pos[0] >= 0 and pos[0] < self.map_size and pos[1] >= 0 and pos[1] < self.map_size
    
    def summon_enemy(self, t, start_id):
        logger.debug('B', 'summon_enemy: {}, {}', t, start_id)
        e = _create_enemy(t, self.start[start_id])
        if self.cost_atk < e.cost:
            logger.verbose('B', 'Summon enemy {} failed due to cost shortage', t)
            return False
        if self.cnt[self.start[start_id][0], self.start[start_id][1], t] >= hyper_parameters.enemy_overlay_limit:
            logger.verbose('B', 'Summon enemy {} failed due to the limit of overlay unit', t)
            return False
        logger.debug('B', 'Summon enemy {}', t)
        self.enemies.append(e)
        self.cost_atk -= e.cost
        self.cnt[self.start[start_id][0], self.start[start_id][1], t] += 1
        return True
    
    def summon_cluster(self, types, start_id):
        logger.debug('B', 'summon_cluster: {}, {}', types, start_id)
        enemies = []
        costs = []
        for t in types:
            if t == 3:
                continue
            e = _create_enemy(t, self.start[start_id])
            costs.append(e.cost)
            enemies.append(e)
        if self.cost_atk < sum(costs):
            logger.verbose('B', 'Summon cluster {} failed due to cost shortage', types)
            return False
        ut, ucnt = np.unique(types, return_counts=True)
        for t, n in zip(ut, ucnt):
            if t == 3:
                continue
            cnt = self.cnt[self.start[start_id][0], self.start[start_id][1], t]
            if cnt + n > hyper_parameters.enemy_overlay_limit:
                logger.verbose(
                    'B',
                    'Summon cluster {} failed due to the limit of overlay unit of type {} ({}+{}/{})',
                    types, t, cnt, n, hyper_parameters.enemy_overlay_limit
                )
                return False
        self.enemies += enemies
        self.cost_atk -= sum(costs)
        for t, n in zip(ut, ucnt):
            if t == 3:
                continue
            self.cnt[self.start[start_id][0], self.start[start_id][1], t] += n
        return True

    def tower_build(self, loc):
        t = _create_tower(loc)
        if self.cost_def < t.cost:
            logger.verbose('B', 'Build tower ({},{}) failed due to cost shortage', loc[0], loc[1])
            return False
        if self.map[loc[0], loc[1], 0] == 1:
            logger.verbose('Cannot overlay tower at ({},{})', loc[0], loc[1])
            return False
        logger.debug('B', 'Build tower ({},{})', loc[0], loc[1])
        self.towers.append(t)
        self.cost_def -= t.cost
        return True

    def tower_atkup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.atklv >= len(config.tower_ATKUp_cost):
                    logger.verbose('B', 'Tower ATK up ({},{}) failed due to atk lv max', loc[0], loc[1])
                    return False
                cost = config.tower_ATKUp_cost[t.atklv]
                atkup = config.tower_ATKUp_list[t.atklv]
                if self.cost_def < cost:
                    logger.verbose('B', 'Tower ATK up ({},{}) failed due to cost shortage', loc[0], loc[1])
                    return False
                logger.debug('B', 'Tower ATK up ({},{})', loc[0], loc[1])
                self.cost_def -= cost
                t.atkUp(atkup, cost)
                return True
        logger.verbose('No tower at ({},{})', loc[0], loc[1])
        return False

    def tower_rangeup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.rangelv >= len(config.tower_rangeUp_cost):
                    logger.verbose('B', 'Tower range up ({},{}) failed due to range lv max', loc[0], loc[1])
                    return False
                cost = config.tower_rangeUp_cost[t.rangelv]
                rangeup = config.tower_rangeUp_list[t.rangelv]
                if self.cost_def < cost:
                    logger.verbose('B', 'Tower range up ({},{}) failed due to cost shortage', loc[0], loc[1])
                    return False
                logger.debug('B', 'Tower range up ({},{})', loc[0], loc[1])
                self.cost_def -= cost
                t.rangeUp(rangeup, cost)
                return True
        logger.verbose('B', 'No tower at ({},{})', loc[0], loc[1])
        return False

    def tower_destruct(self, loc):
        for t in self.towers:
            if loc == t.loc:
                self.cost_def += t.cost * config.tower_destruct_return
                self.towers.remove(t)
                logger.debug('B', 'Destruct tower ({},{})', loc[0], loc[1])
                return True
        logger.verbose('B', 'No tower at ({},{})', loc[0], loc[1])
        return False
    
    def step(self):
        # Forward the game one time step.
        # Return the reward for the defender.
        def dist(a, b):
            return abs(a[0]-b[0])+abs(a[1]-b[1])
        reward = 0.
        reward += config.reward_time
        self.steps += 1
        logger.debug('B', 'Step: {}->{}', self.steps-1, self.steps)
        for t in self.towers:
            t.cd -= 1
            if t.cd > 0: # attack cool down
                continue
            attackable = []
            for e in self.enemies:
                if dist(e.loc, t.loc) <= t.rge:
                    attackable.append(e)
            attackable.sort(key=lambda x: self.map[x.loc[0], x.loc[1]][4] - x.margin)
            if len(attackable) > 0:
                e = attackable[0]
                e.damage(t.atk)
                t.cd = config.tower_attack_interval
                logger.debug('B',
                    'Attack: ({},{})->({},{},{},{})',
                    t.loc[0], t.loc[1], e.loc[0], e.loc[1], e.margin, e.type
                )
                if not e.alive:
                    logger.debug('B', 'Kill')
                    self.enemies.remove(e)
                    reward += config.reward_kill

        dp = [[0,1], [0,-1], [1,0], [-1,0]]
        toremove = []
        for e in self.enemies:
            e.margin += e.speed
            while e.margin >= 1.:
                # move to next grid
                e.margin -= 1.
                d = self.map[e.loc[0], e.loc[1], 5]
                p = [
                    e.loc[0] + dp[d][0],
                    e.loc[1] + dp[d][1]
                ]
                self.cnt[e.loc[0], e.loc[1], e.type] -= 1
                e.setLoc(p)
                self.cnt[p[0], p[1], e.type] += 1
                if e.loc == self.end:
                    reward -= config.penalty_leak
                    logger.debug('B', 'Leak {}', e.type)
                    toremove.append(e)
                    if self.base_LP is not None:
                        logger.debug('B', 'LP: {} -> {}', self.base_LP, max(self.base_LP-1, 0))
                        self.base_LP = max(self.base_LP-1, 0)
                    break
        for e in toremove:
            self.enemies.remove(e)
            self.cnt[e.loc[0], e.loc[1], e.type] -= 1
        
        self.cost_atk = min(self.cost_atk+config.attacker_cost_rate, self.max_cost)
        self.cost_def = min(self.cost_def+config.defender_cost_rate, self.max_cost)
        logger.debug('B', 'Reward: {}', reward)
        return reward
    
    def done(self):
        return (self.base_LP is not None and self.base_LP <= 0) \
            or self.steps >= hyper_parameters.max_episode_steps
    
    def render(self, mode):
        screen_width = 600
        screen_height = 600
        cost_bar_height = 15
        if self.viewer is None: #draw permanent elements here
            self.viewer = rendering.Viewer(screen_width, screen_height+cost_bar_height)
            # create viewer
            background = rendering.FilledPolygon([
                (0, 0), (0, screen_height),
                (screen_width, screen_height), (screen_width, 0)
            ])
            background.set_color(1,1,1)
            self.viewer.add_geom(background)
            # draw background
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
                    if self.map[r,c,0] == 1:
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
        left, right, top, bottom = \
            0, int(screen_width*(self.cost_atk/self.max_cost)//2), \
            screen_height+cost_bar_height, screen_height
        self.viewer.draw_polygon(
            v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ],
            color=(1, .3, 0)
        )
        left, right, top, bottom = \
            screen_width//2, int(screen_width*(1+self.cost_def/self.max_cost)//2), \
            screen_height+cost_bar_height, screen_height
        self.viewer.draw_polygon(
            v=[
                (left, bottom), (left, top),
                (right, top), (right, bottom)
            ],
            color=(0, .3, 1)
        )
        block_width = screen_width // self.map_size
        block_height = screen_height // self.map_size
        for e in self.enemies:
            left, right, top, bottom = \
                screen_width * e.loc[1] // self.map_size, \
                screen_width * (e.loc[1]+1) // self.map_size, \
                screen_height * (e.loc[0]+1) // self.map_size, \
                screen_height * e.loc[0] // self.map_size
            lbheight = screen_height * e.loc[0] // self.map_size
            lblen = int(e.LP / e.maxLP * 3 / 8 * block_width)
            #LP bar
            if e.type == 0: #left up
                left += block_width // 16
                right -= block_width * 9 // 16
                top -= block_height // 24
                bottom += block_height * 7 // 12
                color = (0.6, 0.2, 0.2)
                lbheight += block_height * 13 // 24
            elif e.type == 1: #right up
                left += block_width * 9 // 16
                right -= block_width // 16
                top -= block_height // 24
                bottom += block_height * 7 // 12
                color = (0.2, 0.6, 0.2)
                lbheight += block_height * 13 // 24
            elif e.type == 2: #bottom center
                left += block_width * 5 // 16
                right -= block_width * 5 // 16
                top -= block_height * 13 // 24
                bottom += block_height // 12
                color = (0.2, 0.2, 0.6)
                lbheight += block_height // 24
            self.viewer.draw_polygon(
                v = [
                    (left, bottom), (left, top),
                    (right, top), (right, bottom)
                ],
                color = color
            )
            self.viewer.draw_line(
                (left, lbheight), (left + lblen, lbheight),
                color = (0, 1, 0)
            )
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
                color = (.2, .8, .8)
            )
            self.viewer.draw_polygon(
                v=[
                    (left+block_width//5, bottom),
                    (left+block_width//5, bottom+block_height//3*(t.atklv+1)),
                    (left+block_width*2//5, bottom+block_height//3*(t.atklv+1)),
                    (left+block_width*2//5, bottom),
                ],
                color = (0, 1, 0)
            )
            self.viewer.draw_polygon(
                v=[
                    (left+block_width*3//5, bottom),
                    (left+block_width*3//5, bottom+block_height//3*(t.rangelv+1)),
                    (left+block_width*4//5, bottom+block_height//3*(t.rangelv+1)),
                    (left+block_width*4//5, bottom),
                ],
                color = (0, 1, 0)
            )
            
        if config.base_LP is not None:
            left, bottom = \
                screen_width * self.end[1] // self.map_size, \
                screen_height * self.end[0] // self.map_size
            top = bottom + screen_height // self.map_size // 10
            right = left + screen_width // self.map_size * (self.base_LP / config.base_LP)
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
        if self.viewer:
            self.viewer.close()
            self.viewer = None
