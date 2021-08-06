import json

class Config(object):
    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

def load_config(config):
    with open(config, 'r') as f:
        dict = json.load(f)
    return Config(dict)

def get_device(config):
    import torch
    if torch.cuda.is_available() and not config.debug:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    setattr(config, 'device', device)
    return device
