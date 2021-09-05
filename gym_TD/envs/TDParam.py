class Config(object):
    def __init__(self):
        self.max_enemy_lv = 1
        self.max_tower_lv = 1

        self.enemy_types = 4
        self.tower_types = 4

        self.enemy_LP = [
            [820, 1700],
            [2050, 3000],
            [6000, 10000],
            [8000, 12000]
        ]

        self.enemy_speed = [
            [.25, .25],
            [.13, .13],
            [.1, .1],
            [.1, .1]
        ]

        self.enemy_defense = [
            [0, 0],
            [250, 300],
            [600, 900], # [800, 1000]
            [120, 200] # [80, 100]
        ]

        self.enemy_cost = [
            [5, 5],
            [15, 15],
            [40, 40],
            [30, 30]
        ]

        self.tower_attack = [
            [337, 454],
            [819, 1036], # [546, 691]
            [566, 691],
            [348, 448]
        ]

        self.tower_range = [
            [4, 4],
            [3, 3],
            [5, 5],
            [3, 3]
        ]

        self.tower_splash_range = [
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 0]
        ]

        self.tower_cost = [
            [10, 10],
            [17, 17],
            [23, 23],
            [12, 12]
        ]

        self.tower_attack_interval = [
            [6, 6], # [5, 5]
            [12, 12], # [8, 8]
            [8, 8], # [14, 14]
            [9.5, 9.5]
        ]

        self.tower_destruct_return = .5

        self.frozen_time = 4
        self.frozen_ratio = .2

        self.attacker_init_cost = 10
        self.defender_init_cost = 10
        self.base_LP = 5
        self.max_cost = 100

        self.reward_kill = 0.1
        self.penalty_leak = 10.
        self.reward_time = 0.001

        self.attacker_cost_init_rate = .5
        self.attacker_cost_final_rate = 1
        self.defender_cost_rate = .2

        self.tower_distance = 2
        self.enemy_upgrade_at = 0.75

        self.attacker_action_interval = 1
        self.defender_action_interval = 1

config = Config()

def paramConfig(**kwargs):
    for key, val in kwargs.items():
        setattr(config, key, val)

def getConfig():
    return config.__dict__

class HyperParameters(object):
    def __init__(self):
        super(HyperParameters, self).__setattr__('max_episode_steps', 800)
        super(HyperParameters, self).__setattr__('video_frames_per_second', 50)
        super(HyperParameters, self).__setattr__('allow_multiple_actions', False)
        super(HyperParameters, self).__setattr__('max_cluster_length', 10)
        super(HyperParameters, self).__setattr__('max_num_of_roads', 3)
    def __setattr__(self, name: str, value) -> None:
        raise RuntimeError('You are not supposed to modify hyper parameters during runtime.')

hyper_parameters = HyperParameters()

def getHyperParameters():
    return hyper_parameters.__dict__.copy()
