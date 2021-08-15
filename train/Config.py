import json
from gym_TD import logger

class Config(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
    
    def __str__(self):
        res = '{'
        for k, v in self.__dict__.items():
            res += '{}: {}, '.format(k, v)
        res = res[:-2] + '}'
        return res

def load_config(config):
    with open(config, 'r') as f:
        dict = json.load(f)
    return Config(dict)

def get_device(config):
    import torch
    if torch.cuda.is_available() and config.use_cuda:
        device = torch.device("cuda")
        logger.info('C', "Using CUDA")
    else:
        device = torch.device("cpu")
        logger.info('C', "Using CPU")

    setattr(config, 'device', device)
    return device
