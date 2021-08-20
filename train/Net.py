import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.nn.modules.activation import LogSoftmax

class View(nn.Module):
    # https://github.com/pytorch/vision/issues/720
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class FCN(nn.Module):
    def __init__(
            self,
            cin, h, w,
            prob_out,
            value_out,
            kernels=[5,5,3],
            channels=[256,512,1024],
            pools=[False,True,False],
            prob_channel = 1
        ):
        assert len(kernels) == len(channels) == len(pools), 'FCN: the length of kernels, channels and pools must be the same (input: kernels: {}, channels: {}, pools: {})'.format(kernels, channels, pools)
        super(FCN, self).__init__()

        layers = []
        lastc = cin
        for k, c, p in zip(kernels, channels, pools):
            layers.append(nn.Conv2d(lastc, c, k, padding='same'))
            if p:
                layers.append(nn.MaxPool2d(2))
                h, w = h//2, w//2
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(c))
            lastc = c
        
        layers.append(nn.MaxPool2d([h, w]))
        layers.append(nn.Flatten())

        self.shared = nn.Sequential(*layers)

        if prob_out is not None:
            self.prob_model = nn.Sequential(
                nn.Linear(channels[-1], np.prod(prob_out)),
                View([-1, *prob_out]),
                nn.LogSoftmax(prob_channel)
            )
        
        if value_out is not None:
            self.value_model = nn.Sequential(
                nn.Linear(channels[-1], np.prod(value_out)),
                View([-1, *value_out])
            )

    def forward(self, x):
        x = self.shared(x)
        prob_model = getattr(self, 'prob_model', None)
        value_model = getattr(self, 'value_model', None)
        ret = []
        if prob_model is not None:
            ret.append(prob_model(x))
        if value_model is not None:
            ret.append(value_model(x))

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
class FullyConnected(nn.Module):
    def __init__(
        self,
        input_shape,
        prob_out, value_out,
        hiddens, prob_channel = 1
    ):
        super().__init__()
        layers = [
            nn.Flatten(),
        ]
        lastn = np.prod(input_shape)
        for h in hiddens:
            layers.append(nn.Linear(lastn, h))
            layers.append(nn.ReLU())
            lastn = h
        self.shared = nn.Sequential(*layers)

        if prob_out is not None:
            self.prob_model = nn.Sequential(
                nn.Linear(lastn, np.prod(prob_out)),
                View([-1, *prob_out]),
                nn.LogSoftmax(prob_channel)
            )
        if value_out is not None:
            self.value_model = nn.Sequential(
                nn.Linear(lastn, np.prod(value_out)),
                View([-1, *value_out])
            )
    
    def forward(self, x):
        x = self.shared(x)
        prob_model = getattr(self, 'prob_model', None)
        value_model = getattr(self, 'value_model', None)
        ret = []
        if prob_model is not None:
            ret.append(prob_model(x))
        if value_model is not None:
            ret.append(value_model(x))

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

class UNet(nn.Module):
    def __init__(
            self,
            cin, ccomp, h, w,
            prob_out,
            value_out,
            kernels=[3,3,3,1],
            channels=[64,128,256,512],
            extra_act=1,
            value_type='independent',
            prob_channel = 1
        ):
        assert len(kernels) == len(channels), 'UNet: the length of kernels and channels must be the same (input: kernels: {}, channels: {})'.format(kernels, channels)
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

        self.prob_out = prob_out
        self.value_out = value_out

        if ccomp > 0:
            self.conv0 = nn.Conv2d(cin, ccomp, 1)
            lastc = ccomp
        else:
            self.conv0 = nn.Identity()
            lastc = cin

        for i, (k, c) in enumerate(zip(kernels[:-1], channels[:-1])):
            down_conv(i+1, k, lastc, c)
            up_conv(i+1, k, channels[i+1], c, c, [h%2, w%2])
            h, w = h//2, w//2
            lastc = c

        self.conv_bottom = nn.Conv2d(lastc, channels[-1], kernels[-1], padding='same')

        self.nlayers = len(kernels)

        if prob_out is not None:
            self.prob_conv = nn.Conv2d(channels[0], prob_out, 1)
            self.prob_flat = nn.Flatten()
            if extra_act > 0:
                self.prob_dense = nn.Linear(prob_out*oh*ow, extra_act)
            self.logsoftmax = nn.LogSoftmax(prob_channel)
        
        if value_out is not None:
            if value_type == 'independent':
                self.value_layer = nn.Sequential(
                    nn.Conv2d(channels[0], value_out, 1),
                    nn.MaxPool2d([oh, ow]),
                    nn.Flatten(),
                    nn.Linear(value_out, value_out)
                )
            elif value_type == 'dependent':
                self.value_conv = nn.Conv2d(channels[0], value_out, 1)
                self.value_flat = nn.Flatten()
                if extra_act > 0:
                    self.value_dense = nn.Linear(value_out*oh*ow, extra_act)
                    
    
    def forward(self, x):
        x = func.relu(self.conv0(x))
        v = {}
        for i in range(1, self.nlayers):
            x = func.relu(getattr(self, 'conv{}_1'.format(i))(x))
            x = func.relu(getattr(self, 'conv{}_2'.format(i))(x))
            v['c{}'.format(i)] = x
            x = getattr(self, 'maxpool{}'.format(i))(x)
            x = getattr(self, 'bn{}'.format(i))(x)
        x = func.relu(self.conv_bottom(x))
        for i in reversed(range(1, self.nlayers)):
            x = func.relu(getattr(self, 'up{}'.format(i))(x))
            x = torch.cat([x, v['c{}'.format(i)]], 1)
            x = getattr(self, 'upbn{}'.format(i))(x)
            x = func.relu(getattr(self, 'upconv{}_1'.format(i))(x))
            x = func.relu(getattr(self, 'upconv{}_2'.format(i))(x))
        
        ret = []
        if self.prob_out is not None:
            prob = self.prob_conv(x)
            prob = self.prob_flat(prob)
            if getattr(self, 'prob_dense', None) is not None:
                nop = self.prob_dense(prob)
                prob = torch.cat([prob, nop], 1)
            prob = self.logsoftmax(prob)
            ret.append(prob)
        if self.value_out is not None:
            value_layer = getattr(self, 'value_layer', None)
            if value_layer is not None:
                v = self.value_layer(x)
                ret.append(v)
            else:
                v = self.value_conv(x)
                v = self.value_flat(x)
                if getattr(self, 'value_dense', None) is not None:
                    nop = self.value_dense(v)
                    v = torch.cat([v, nop], 1)
                ret.append(v)
        
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

class NetWrapper(object):
    def __init__(self, module, idx):
        super().__init__()
        self.module = module
        self.idx = idx
    
    def forward(self, x):
        ret = self.module(x)
        return ret[self.idx]
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.module, name)
