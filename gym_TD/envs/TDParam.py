class Config(object):
    def __init__(self):
        self.enemy_balance_LP = 10
        self.enemy_balance_speed = 2
        self.enemy_balance_cost = 3

        self.enemy_strong_LP = 20
        self.enemy_strong_speed = 1
        self.enemy_strong_cost = 3

        self.enemy_fast_LP = 5
        self.enemy_fast_speed = 4
        self.enemy_fast_cost = 3

        self.attacker_init_cost = 10
        self.defender_init_cost = 5

        self.tower_basic_cost = 5
        self.tower_destruct_return = 0.5

        self.tower_basic_ATK = 5
        self.tower_ATKUp_list = [5, 5]
        self.tower_ATKUp_cost = [5, 5]

        self.tower_basic_range = 4
        self.tower_rangeUp_list = [2, 2]
        self.tower_rangeUp_cost = [3, 3]

        self.reward_kill = 0.1
        self.penalty_leak = 1.
        self.reward_time = 0.001


config = Config()

def paramConfig(**kwargs):
    for key, val in kwargs.items():
        setattr(config, key, val)
