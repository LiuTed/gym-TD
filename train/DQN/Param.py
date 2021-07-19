import torch
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE = 2e-3
GAMMA = 0.99
MEMORY_SIZE = 50000
UPDATE = 200
MAP_SIZE = 20
NUM_EPISODE = 100
STEPS_PER_EPISODE = 10000
DEBUG = False
TEST_EPISODE = 20
DO_TEST_EVERY_LOOP = 80
STATE_BASED = True

if torch.cuda.is_available() and not DEBUG:
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")
