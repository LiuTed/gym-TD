import torch
import numpy as np

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GAMMA = 0.9
MEMORY_SIZE = 10000
UPDATE = 200
MAP_SIZE = 20
NUM_EPISODE = 100
STEPS_PER_EPISODE = 1000
DEBUG = False
TEST_EPISODE = 20
DO_TEST_EVERY_LOOP = 20
STATE_BASED = True

if torch.cuda.is_available() and not DEBUG:
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
