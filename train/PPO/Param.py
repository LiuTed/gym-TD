import torch
import numpy as np

BATCH_SIZE = 64
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 2e-4
GAMMA = 0.9
MEMORY_SIZE = 10000
UPDATE = 200
MAP_SIZE = 20
NUM_EPISODE = 100
STEPS_PER_EPISODE = 10000
DEBUG = True
TEST_EPISODE = 20
DO_TEST_EVERY_LOOP = 80
STATE_BASED = True
PPO_VERSION = 2
ACTOR_UPDATE_LOOP = 10
CRITIC_UPDATE_LOOP = 10

if torch.cuda.is_available() and not DEBUG:
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
