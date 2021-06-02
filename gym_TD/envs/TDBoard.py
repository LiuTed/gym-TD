import numpy as np

from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym_TD.envs.TDParam import config

class Enemy(object):
    def __init__(self, maxLP, speed, cost, loc, t=None):
        self.maxLP = self.LP = maxLP
        self.speed = speed
        self.cost = cost
        self.loc = loc
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
    channels = {
        "road": 0,
        "start": 1,
        "end": 2,
        "tower": 3,
        "atklv": 4,
        "rangelv": 5,
        "enemy0LP": 6,
        "enemy1LP": 7,
        "enemy2LP": 8
    }
    def __init__(self, map_size, np_random, cost_def, cost_atk, max_cost):
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

        self.map = np.empty(shape=(map_size, map_size, 2), dtype=np.int32)
        # if road, dist to endpoint
        road = self.create_road(np_random)
        dist = 0
        for p in road:
            self.map[p[0], p[1]][0] = 1
            self.map[p[0], p[1]][1] = dist
            dist += 1

        self.enemies = []
        self.towers = []
        self.cost_def = cost_def
        self.cost_atk = cost_atk
        self.max_cost = max_cost

        self.viewer = None

        self.steps = 0
    
    def get_states(self):
        s = np.zeros(shape=(self.map_size, self.map_size, 9), dtype=np.float32)
        for r in range(self.map_size):
            for c in range(self.map_size):
                if self.map[r,c,0] == 1:
                    s[r,c,0] = 1
        s[self.start[0], self.start[1]][1] = 1
        s[self.end[0], self.end[1]][2] = 1
        for t in self.towers:
            s[t.loc[0], t.loc[1]][3] = 1
            s[t.loc[0], t.loc[1]][4] = t.atklv
            s[t.loc[0], t.loc[1]][5] = t.rangelv
        for e in self.enemies:
            s[e.loc[0], e.loc[1]][e.type+6] = e.LP/e.maxLP
        return s
    
    def summon_enemy(self, t):
        e = _create_enemy(t, self.start)
        if self.cost_atk < e.cost:
            return
        self.enemies.append(e)
        self.cost_atk -= e.cost
    def tower_build(self, loc):
        t = _create_tower(loc)
        if self.cost_def < t.cost:
            return
        if self.map[loc[0], loc[1], 0] == 1:
            return
        self.towers.append(t)
        self.cost_def -= t.cost
    def tower_atkup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.atklv >= len(config.tower_ATKUp_cost):
                    break
                cost = config.tower_ATKUp_cost[t.atklv]
                atkup = config.tower_ATKUp_list[t.atklv]
                if self.cost_def < cost:
                    break
                self.cost_def -= cost
                t.atkUp(atkup, cost)
                break
    def tower_rangeup(self, loc):
        for t in self.towers:
            if loc == t.loc:
                if t.rangelv >= len(config.tower_rangeUp_cost):
                    break
                cost = config.tower_rangeUp_cost[t.rangelv]
                rangeup = config.tower_rangeUp_list[t.rangelv]
                if self.cost_def < cost:
                    break
                self.cost_def -= cost
                t.rangeUp(rangeup, cost)
                break
    def tower_destruct(self, loc):
        for t in self.towers:
            if loc == t.loc:
                self.cost_def += int(t.cost * config.tower_destruct_return)
                self.towers.remove(t)
                break
    
    def step(self):
        # Forward the game one time step.
        # Return the reward for the defender.
        def dist(a, b):
            return abs(a[0]-b[0])+abs(a[1]-b[1])
        reward = 0.
        reward += config.reward_time
        self.steps += 1
        for t in self.towers:
            attackable = []
            for e in self.enemies:
                if dist(e.loc, t.loc) <= t.rge:
                    attackable.append(e)
            attackable.sort(key=lambda x: self.map[x.loc[0], x.loc[1]][1])
            if len(attackable) > 0:
                e = attackable[0]
                e.damage(t.atk)
                if not e.alive:
                    self.enemies.remove(e)
                    reward += config.reward_kill
        dp = [[0,1], [0,-1], [1,0], [-1,0]]
        toremove = []
        for e in self.enemies:
            for _ in range(e.speed):
                d = self.map[e.loc[0], e.loc[1], 1]
                for i in range(4):
                    p = [
                        e.loc[0] + dp[i][0],
                        e.loc[1] + dp[i][1]
                    ]
                    if p[0] < 0 or p[0] >= self.map_size \
                        or p[1] < 0 or p[1] >= self.map_size:
                        continue
                    if self.map[p[0], p[1], 0] == 1 and self.map[p[0], p[1], 1] < d:
                        e.setLoc(p)
                        break
                if e.loc == self.end:
                    reward -= config.penalty_leak
                    toremove.append(e)
        for e in toremove:
            self.enemies.remove(e)
        
        self.cost_atk = min(self.cost_atk+1, self.max_cost)
        self.cost_def = min(self.cost_def+1, self.max_cost)
        return reward

    def create_road(self, np_random):
        # Return a random list of points from end point to start point.
        class Point(object):
            def __init__(self, pos, dist, prev=None):
                self.pos = pos
                self.dist = dist
                self.prev = prev
        _dp = [[0,1], [0,-1], [1,0], [-1,0]]

        self.height = np_random.randint(low=1, high=100, size=(self.map_size, self.map_size))
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
                if p.dist + self.height[cord[0], cord[1]] \
                     < dist[cord[0], cord[1]]:
                    dist[cord[0], cord[1]] = \
                        p.dist + self.height[cord[0], cord[1]]
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
            background = rendering.FilledPolygon([
                (0, 0), (0, screen_height),
                (screen_width, screen_height), (screen_width, 0)
            ])
            background.set_color(1,1,1)
            self.viewer.add_geom(background)
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

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
