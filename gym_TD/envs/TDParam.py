class Config(object):
    def __init__(self):
        self.max_enemy_lv = 1
        self.max_tower_lv = 1

        self.enemy_types = 4
        self.tower_types = 4

        self.enemy_LP = [
            [820, 1700],
            [2050, 3000],
            [6000, 8000], # [6000, 10000],
            [8000, 12000]
        ]

        self.enemy_speed = [
            [.25, .25], # 0.38
            [.13, .13], # 0.2
            [.1, .1], # 0.15
            [.1, .1] # 0.15
        ]

        self.enemy_defense = [
            [0, 0],
            [200, 250], # [250, 300]
            [600, 800], # [800, 1000]
            [80, 100] # [80, 100]
        ]

        self.enemy_cost = [
            [8, 8],
            [15, 15],
            [40, 40],
            [30, 30]
        ]

        self.tower_attack = [
            [454, 540],
            [651, 771], # [543, 643],
            [566, 691],
            [358, 424] # [448, 530]
        ]

        self.tower_range = [
            [3, 3],
            [2, 2],
            [4, 4], # [4, 4]
            [3, 3] # [3, 3]
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
            [12, 12] # [12, 12]
        ]

        self.tower_attack_interval = [
            [2, 2], # [5, 5]
            [4, 4], # [8, 8]
            [7, 7], # [14, 14]
            [4.75, 4.75] # [9.5, 9.5]
        ]

        self.tower_destruct_return = .5

        self.frozen_time = 2
        self.frozen_ratio = .2

        self.attacker_init_cost = 0
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
        super(HyperParameters, self).__setattr__('max_episode_steps', 1200)
        super(HyperParameters, self).__setattr__('video_frames_per_second', 50)
        super(HyperParameters, self).__setattr__('allow_multiple_actions', False)
        super(HyperParameters, self).__setattr__('max_cluster_length', 1)
        super(HyperParameters, self).__setattr__('max_num_of_roads', 3)
    def __setattr__(self, name: str, value) -> None:
        raise RuntimeError('You are not supposed to modify hyper parameters during runtime.')

hyper_parameters = HyperParameters()

def getHyperParameters():
    return hyper_parameters.__dict__.copy()
