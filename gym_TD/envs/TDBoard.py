import numpy as np

from gym import logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym_TD.envs.TDParam import config, hyper_parameters

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
    def __init__(self, map_size, np_random, cost_def, cost_atk, max_cost, base_LP):
        self.map_size = map_size
        start_edge = np_random.randint(low=0, high=4) #start point at which edge
        start_idx = np_random.randint(low=0, high=map_size-1) #location of start point
        end_idx = np_random.randint(low=0, high=map_size-1) #location of end point
        idx2point = [
            lambda i: [0, i], #top
            lambda i: [i, map_size-1], #right
            lambda i: [map_size-1-i, 0], #left
            lambda i: [map_size-1, map_size-1-i] #bottom
        ] # map index to coordinate
        self.start = idx2point[start_edge](start_idx)
        self.end = idx2point[3-start_edge](end_idx)
        # end point at the other side of start point

        self.map = np.zeros(shape=(map_size, map_size, 3), dtype=np.int32)
        # if road, dist to end, where to go
        road = self.create_road(np_random)
        last = None
        for p in reversed(road): # from start to end
            self.map[p[0], p[1], 0] = 1
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
                self.map[last[0], last[1], 2] = direct
            last = p
        dist = 0
        for p in road: # from end to start
            self.map[p[0], p[1], 1] = dist
            dist += 1
        
        logger.debug('Map: ' + str(self.map))

        self.cnt = np.zeros(shape=(self.map_size, self.map_size, 3), dtype=np.int32)
        # how many enemies of type [2] at location ([0], [1])

        self.enemies = []
        self.towers = []
        self.cost_def = cost_def
        self.cost_atk = cost_atk
        self.max_cost = max_cost
        self.base_LP = base_LP

        self.viewer = None

        self.steps = 0
    
    def get_states(self):
        ptr = np.zeros(shape=(self.map_size, self.map_size, 3), dtype=np.int32)
        s = np.zeros(shape=self.state_shape, dtype=np.float32)
        s[:,:,0] = self.map[:,:,0]
        s[self.start[0], self.start[1], 1] = 1
        s[self.end[0], self.end[1], 2] = 1
        s[:,:,3] = np.full(shape=(self.map_size, self.map_size), fill_value=self.cost_def/self.max_cost, dtype=np.float32)
        s[:,:,4] = np.full(shape=(self.map_size, self.map_size), fill_value=self.cost_atk/self.max_cost, dtype=np.float32)
        for t in self.towers:
            s[t.loc[0], t.loc[1], 5] = 1
            s[t.loc[0], t.loc[1], 6] = t.atklv
            s[t.loc[0], t.loc[1], 7] = t.rangelv
        for e in self.enemies:
            n = ptr[e.loc[0], e.loc[1], e.type]
            s[e.loc[0], e.loc[1], 8+hyper_parameters.enemy_overlay_limit*e.type+n] = e.LP/e.maxLP
            s[e.loc[0], e.loc[1], 8+hyper_parameters.enemy_overlay_limit*(e.type+3)+n] = e.margin
            ptr[e.loc[0], e.loc[1], e.type] += 1
        return s
    
    @staticmethod
    def n_channels():
        return 9 + hyper_parameters.enemy_overlay_limit * 6
    @property
    def state_shape(self):
        return (self.map_size, self.map_size, self.n_channels())
    
    def summon_enemy(self, t):
        e = _create_enemy(t, self.start)
        if self.cost_atk < e.cost:
            logger.debug('Summon enemy {} failed due to cost shortage'.format(t))
            return False
        if self.cnt[self.start[0], self.start[1], t] >= hyper_parameters.enemy_overlay_limit:
            logger.debug('Summon enemy {} failed due to the limit of overlay unit'.format(t))
            return False
        logger.debug('Summon enemy {}'.format(t))
        self.enemies.append(e)
        self.cost_atk -= e.cost
        self.cnt[self.start[0], self.start[1], t] += 1
        return True

    def tower_build(self, loc):
        t = _create_tower(loc)
        if self.cost_def < t.cost:
            logger.debug('Build tower ({},{}) failed due to cost shortage'.format(loc[0], loc[1]))
            return False
        if self.map[loc[0], loc[1], 0] == 1:
            logger.debug('Cannot overlay tower at ({},{})'.format(loc[0], loc[1]))
            return False
        logger.debug('Build tower ({},{})'.format(loc[0], loc[1]))
        self.towers.append(t)
        self.cost_def -= t.cost
        return True

    def tower_atkup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.atklv >= len(config.tower_ATKUp_cost):
                    logger.debug('Tower ATK up ({},{}) failed due to atk lv max'.format(loc[0], loc[1]))
                    return False
                cost = config.tower_ATKUp_cost[t.atklv]
                atkup = config.tower_ATKUp_list[t.atklv]
                if self.cost_def < cost:
                    logger.debug('Tower ATK up ({},{}) failed due to cost shortage'.format(loc[0], loc[1]))
                    return False
                logger.debug('Tower ATK up ({},{})'.format(loc[0], loc[1]))
                self.cost_def -= cost
                t.atkUp(atkup, cost)
                return True
        logger.debug('No tower at ({},{})'.format(loc[0], loc[1]))
        return False

    def tower_rangeup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.rangelv >= len(config.tower_rangeUp_cost):
                    logger.debug('Tower range up ({},{}) failed due to range lv max'.format(loc[0], loc[1]))
                    return False
                cost = config.tower_rangeUp_cost[t.rangelv]
                rangeup = config.tower_rangeUp_list[t.rangelv]
                if self.cost_def < cost:
                    logger.debug('Tower range up ({},{}) failed due to cost shortage'.format(loc[0], loc[1]))
                    return False
                logger.debug('Tower range up ({},{})'.format(loc[0], loc[1]))
                self.cost_def -= cost
                t.rangeUp(rangeup, cost)
                return True
        logger.debug('No tower at ({},{})'.format(loc[0], loc[1]))
        return False

    def tower_destruct(self, loc):
        for t in self.towers:
            if loc == t.loc:
                self.cost_def += t.cost * config.tower_destruct_return
                self.towers.remove(t)
                logger.debug('Destruct tower ({},{})'.format(loc[0], loc[1]))
                return True
        logger.debug('No tower at ({},{})'.format(loc[0], loc[1]))
        return False
    
    def step(self):
        # Forward the game one time step.
        # Return the reward for the defender.
        def dist(a, b):
            return abs(a[0]-b[0])+abs(a[1]-b[1])
        reward = 0.
        reward += config.reward_time
        self.steps += 1
        logger.debug('Step: {}->{}'.format(self.steps-1, self.steps))
        for t in self.towers:
            t.cd -= 1
            if t.cd > 0: # attack cool down
                continue
            attackable = []
            for e in self.enemies:
                if dist(e.loc, t.loc) <= t.rge:
                    attackable.append(e)
            attackable.sort(key=lambda x: self.map[x.loc[0], x.loc[1]][1] - x.margin)
            if len(attackable) > 0:
                e = attackable[0]
                e.damage(t.atk)
                t.cd = config.tower_attack_interval
                logger.debug('Attack: ({},{})->({},{},{},{})'.format(t.loc[0], t.loc[1], e.loc[0], e.loc[1], e.margin, e.type))
                if not e.alive:
                    logger.debug('Kill')
                    self.enemies.remove(e)
                    reward += config.reward_kill

        dp = [[0,1], [0,-1], [1,0], [-1,0]]
        toremove = []
        for e in self.enemies:
            e.margin += e.speed
            while e.margin >= 1.:
                # move to next grid
                e.margin -= 1.
                d = self.map[e.loc[0], e.loc[1], 2]
                p = [
                    e.loc[0] + dp[d][0],
                    e.loc[1] + dp[d][1]
                ]
                self.cnt[e.loc[0], e.loc[1], e.type] -= 1
                e.setLoc(p)
                self.cnt[p[0], p[1], e.type] += 1
                if e.loc == self.end:
                    reward -= config.penalty_leak
                    logger.debug('Leak {}'.format(e.type))
                    toremove.append(e)
                    if self.base_LP is not None:
                        logger.debug('LP: {} -> {}'.format(self.base_LP, max(self.base_LP-1, 0)))
                        self.base_LP = max(self.base_LP-1, 0)
                    break
        for e in toremove:
            self.enemies.remove(e)
            self.cnt[e.loc[0], e.loc[1], e.type] -= 1
        
        self.cost_atk = min(self.cost_atk+config.attacker_cost_rate, self.max_cost)
        self.cost_def = min(self.cost_def+config.defender_cost_rate, self.max_cost)
        logger.debug('Reward: {}'.format(reward))
        return reward
    
    def done(self):
        return (self.base_LP is not None and self.base_LP <= 0) \
            or self.steps >= hyper_parameters.max_episode_steps

    def create_road(self, np_random):
        # Return a random list of points from end point to start point.
        class Point(object):
            def __init__(self, pos, dist, prev=None):
                self.pos = pos
                self.dist = dist
                self.prev = prev
        _dp = [[0,1], [0,-1], [1,0], [-1,0]]

        height = np_random.randint(low=1, high=100, size=(self.map_size, self.map_size))
        # height map, used for create actual map

        dist = np.full((self.map_size, self.map_size), 101*self.map_size*self.map_size)
        dist[self.start[0], self.start[1]] = 0
        queue = [Point(self.start, 0)]

        while len(queue) > 0:
            p = queue[0]
            if p.pos == self.end:
                break
            queue.remove(p)
            for i in range(4):
                cord = [
                    p.pos[0] + _dp[i][0],
                    p.pos[1] + _dp[i][1]
                ]
                if cord[0] < 0 or cord[0] >= self.map_size \
                    or cord[1] < 0 or  cord[1] >= self.map_size:
                    continue # out of range
                if p.dist != dist[p.pos[0], p.pos[1]]: #updated
                    continue
                if p.dist + height[cord[0], cord[1]] \
                     < dist[cord[0], cord[1]]:
                    dist[cord[0], cord[1]] = \
                        p.dist + height[cord[0], cord[1]]
                    queue.append(Point(cord, dist[cord[0], cord[1]], p))
                    queue.sort(key=lambda p: p.dist) #Dijkstra
        p = queue[0]
        road = []
        while p.prev is not None:
            road.append(p.pos)
            p = p.prev
        road.append(self.start)
        return road
    
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
            
            left, right, top, bottom = \
                screen_width * self.start[1] // self.map_size, \
                screen_width * (self.start[1]+1) // self.map_size, \
                screen_height * (self.start[0]+1) // self.map_size, \
                screen_height * self.start[0] // self.map_size
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
