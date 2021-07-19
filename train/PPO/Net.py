import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class FCN(nn.Module):
    def __init__(self, cin, h, w, out, out_prob = True):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(cin, 128, 5, padding='same')
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 5, padding='same')
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(256)
        h, w = h//2, w//2

        self.conv3 = nn.Conv2d(256, 512, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 1024, 1)
        self.pool4 = nn.MaxPool2d([h, w])
        # self.dense1 = nn.Linear(self.size3[0]*self.size3[1]*64, 128)
        self.dense2 = nn.Linear(1024, out)

        if out_prob:
            self.activ = nn.Softmax(1)
        else:
            self.activ = nn.Identity()

        self.model = nn.Sequential(
            self.conv1, nn.ReLU(), self.bn1,
            self.conv2, self.pool2, nn.ReLU(), self.bn2,
            self.conv3, nn.ReLU(), self.bn3,
            self.conv4, self.pool4, nn.ReLU(),
            nn.Flatten(),
            self.dense2,
            self.activ
        )

    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self, cin, h, w):
        super(UNet, self).__init__()
        def down_conv(id, ks, in_c, out_c):
            setattr(self, 'conv{}_1'.format(id), nn.Conv2d(in_c, out_c, ks, padding='same'))
            setattr(self, 'conv{}_2'.format(id), nn.Conv2d(out_c, out_c, ks, padding='same'))
            setattr(self, 'maxpool{}'.format(id), nn.MaxPool2d(2))
            setattr(self, 'bn{}'.format(id), nn.BatchNorm2d(out_c))
        def up_conv(id, ks, in1, in2, out_c, pad):
            setattr(self, 'up{}'.format(id), nn.ConvTranspose2d(in1, out_c, 2, 2, output_padding=pad))
            setattr(self, 'upbn{}'.format(id), nn.BatchNorm2d(out_c+in2))
            setattr(self, 'upconv{}_1'.format(id), nn.Conv2d(out_c+in2, out_c, ks, padding='same'))
            setattr(self, 'upconv{}_2'.format(id), nn.Conv2d(out_c, out_c, ks, padding='same'))

        oh, ow = h, w

        self.conv0 = nn.Conv2d(cin, 32, 1)
        down_conv(1, 5, 32, 64)
        up_conv(1, 3, 128, 64, 64, [h%2, w%2])
        h, w = h//2, w//2
        down_conv(2, 3, 64, 128)
        up_conv(2, 3, 256, 128, 128, [h%2, w%2])
        h, w = h//2, w//2
        down_conv(3, 3, 128, 256)
        up_conv(3, 3, 512, 256, 256, [h%2, w%2])
        h, w = h//2, w//2
        self.conv4 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv5 = nn.Conv2d(64, 4, 1)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(4*oh*ow, 1)
    
    def forward(self, x):
        x = func.relu(self.conv0(x))
        v = {}
        for i in [1, 2, 3]:
            x = func.relu(getattr(self, 'conv{}_1'.format(i))(x))
            x = func.relu(getattr(self, 'conv{}_2'.format(i))(x))
            v['c{}'.format(i)] = x
            x = getattr(self, 'maxpool{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
        x = func.relu(self.conv4(x))
        for i in [3, 2, 1]:
            x = func.relu(getattr(self, 'up{}'.format(i))(x))
            x = torch.cat([x, v['c{}'.format(i)]], 1)
            x = getattr(self, 'upbn{}'.format(i))(x)
            x = func.relu(getattr(self, 'upconv{}_1'.format(i))(x))
            x = func.relu(getattr(self, 'upconv{}_2'.format(i))(x))
        x = self.conv5(x)
        x = self.flat(x)
        nop = self.dense(x)
        x = torch.cat([x, nop], 1)
        return func.softmax(x, 1)

