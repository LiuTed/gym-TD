from gym_TD.envs.TDParam import config, hyper_parameters
from gym_TD.utils import logger

class Enemy(object):
    def __init__(self, maxLP, speed, defense, cost, loc, dist, t):
        self.maxLP = self.LP = maxLP
        self.speed = speed
        self.defense = defense
        self.cost = cost
        self.loc = loc
        self.margin = 0.
        self.dist = dist
        self.slowdown = 0
        self.type = t
    
    def move(self, newLoc, newDist):
        self.loc = newLoc
        self.dist = newDist
    def damage(self, atk):
        dmg = max(atk - self.defense, 0)
        if dmg < atk * .05:
            dmg = atk * .05
        self.LP -= dmg
        if self.LP <= 0:
            self.LP = 0
    @property
    def alive(self):
        return self.LP > 0

def create_enemy(t, loc, dist, lv):
    e = Enemy(
        config.enemy_LP[t][lv],
        config.enemy_speed[t][lv],
        config.enemy_defense[t][lv],
        config.enemy_cost[t][lv],
        loc,
        dist,
        t
    )
    return e

class Tower(object):
    def __init__(self, atk, rge, dmgrge, intv, loc, cost, t):
        self.atk = atk
        self.rge = rge
        self.loc = loc
        self.dmgrge = dmgrge
        self.cost = cost
        self.lv = 0
        self.intv = intv
        self.cd = 0
        self.type = t

    def lvup(self, atk, rge, dmgrge, intv, cost):
        self.lv += 1
        self.atk = atk
        self.rge = rge
        self.dmgrge = dmgrge
        self.intv = intv
        self.cost += cost

    def attack(self, enemies):
        pass
    @staticmethod
    def dist(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

class TowerSingle(Tower):
    def attack(self, enemies):
        to_del = []
        for e in enemies:
            if self.dist(e.loc, self.loc) <= self.rge:
                e.damage(self.atk)
                self.cd += self.intv
                if not e.alive:
                    to_del.append(e)
                break
        return to_del

class TowerBomb(Tower):
    def attack(self, enemies):
        to_del = []
        target = None
        for e in enemies:
            if self.dist(e.loc, self.loc) <= self.rge:
                target = e
                self.cd += self.intv
                break
        if target is not None:
            for e in enemies:
                if self.dist(target.loc, e.loc) <= self.dmgrge:
                    e.damage(self.atk)
                    if not e.alive:
                        to_del.append(e)
        return to_del

class TowerFrozen(Tower):
    def attack(self, enemies):
        to_del = []
        for e in enemies:
            if self.dist(e.loc, self.loc) <= self.rge:
                e.damage(self.atk)
                self.cd += self.intv
                if not e.alive:
                    to_del.append(e)
                else:
                    e.slowdown = config.frozen_time
                break
        return to_del

def create_tower(t, loc):
    tower_classes = {
        0: TowerSingle,
        1: TowerSingle,
        2: TowerBomb,
        3: TowerFrozen
    }
    tower = tower_classes[t](
        config.tower_attack[t][0],
        config.tower_range[t][0],
        config.tower_splash_range[t][0],
        config.tower_attack_interval[t][0],
        loc,
        config.tower_cost[t][0],
        t
    )
    return tower

def upgrade_tower(tower):
    if tower.lv >= config.max_tower_lv:
        logger.debug(
            'E',
            'upgrade_tower: tower ({},{}, lv {}, type {}) is not upgradable',
            tower.loc[0], tower.loc[1], tower.lv, tower.type
        )
        return False
    else:
        t = tower.type
        l = tower.lv + 1
        tower.lvup(
            config.tower_attack[t][l],
            config.tower_range[t][l],
            config.tower_splash_range[t][l],
            config.tower_cost[t][l],
            config.tower_attack_interval[t][l]
        )
        return True
