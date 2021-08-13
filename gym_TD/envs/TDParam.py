class Config(object):
    def __init__(self):
        self.enemy_balance_LP = 15
        self.enemy_balance_speed = 0.2
        self.enemy_balance_cost = 3

        self.enemy_strong_LP = 25
        self.enemy_strong_speed = 0.11
        self.enemy_strong_cost = 3

        self.enemy_fast_LP = 8
        self.enemy_fast_speed = 0.32
        self.enemy_fast_cost = 3

        self.attacker_init_cost = 20
        self.defender_init_cost = 10
        self.base_LP = 3
        self.max_cost = 50

        self.tower_basic_cost = 20
        self.tower_destruct_return = 0.5

        self.tower_basic_ATK = 5
        self.tower_ATKUp_list = [5, 5]
        self.tower_ATKUp_cost = [10, 10]

        self.tower_basic_range = 3
        self.tower_rangeUp_list = [2, 2]
        self.tower_rangeUp_cost = [8, 8]

        self.reward_kill = 0.1
        self.penalty_leak = 1.
        self.reward_time = 0.003

        self.tower_attack_interval = 3
        self.attacker_action_interval = 10
        self.defender_action_interval = 1

        self.attacker_cost_rate = 1.
        self.defender_cost_rate = .5

config = Config()

def paramConfig(**kwargs):
    for key, val in kwargs.items():
        setattr(config, key, val)

def getConfig():
    return config.__dict__

class HyperParameters(object):
    def __init__(self):
        super(HyperParameters, self).__setattr__('max_episode_steps', 300)
        super(HyperParameters, self).__setattr__('video_frames_per_second', 50)
        super(HyperParameters, self).__setattr__('enemy_overlay_limit', 25)
        super(HyperParameters, self).__setattr__('allow_multiple_actions', False)
        super(HyperParameters, self).__setattr__('max_cluster_length', 16)
        super(HyperParameters, self).__setattr__('max_num_of_roads', 3)
    def __setattr__(self, name: str, value) -> None:
        raise RuntimeError('You are not supposed to modify hyper parameters during runtime.')

hyper_parameters = HyperParameters()

def getHyperParameters():
    return hyper_parameters.__dict__.copy()
