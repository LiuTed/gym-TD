import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class FCN(nn.Module):
    def __init__(self, h, w, out):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(129, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.size1 = [(h-4), (w-4)]

        self.conv2 = nn.Conv2d(64, 128, 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.size2 = [(self.size1[0]-4), (self.size1[1]-4)]

        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.size3 = [(self.size2[0]-2), (self.size2[1]-2)]

        self.conv4 = nn.Conv2d(256, 512, self.size3)
        # self.dense1 = nn.Linear(self.size3[0]*self.size3[1]*64, 128)
        self.dense2 = nn.Linear(512, out)

        self.layers = [
            self.conv1, self.bn1, func.relu,
            self.conv2, self.bn2, func.relu,
            self.conv3, self.bn3, func.relu,
            self.conv4, func.relu,
            nn.Flatten(),
            self.dense2
        ]
        self.out = out

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


