#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : audio_smodel.py
@Time    : 2024/01/03 13:49:10
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor


# local attention
class LocalAttention(nn.Module):
    """
    inchannels:输入的通道数
    """

    def __init__(self, inchannels):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=3,
                               out_channels=1,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        convout = self.conv1(x)
        out = torch.cat([avgout, maxout, convout], dim=1)
        out = self.sigmoid(self.conv2(out))
        out = out * x
        return out


# basic conv block has conv instancenorm and prelu
class BasicConv(nn.Module):
    """
    conv block :
        conv 
        InstanceNorm2d
        prelu
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=1,
                 groups=1,
                 relu=True,
                 inorm=True,
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.inorm = nn.InstanceNorm2d(out_channels,
                                       affine=True) if inorm else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):

        x = self.conv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


# dense block
class DenseBlock(nn.Module):
    """
    dense block:
        depth 4
        inchannels 64
        step 2
        net layers:
            constantpad
            conv
            instancenorm
            prelu
    """

    def __init__(self, depth=4, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1),
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(
                self, 'conv{}'.format(i + 1),
                nn.Conv2d(self.in_channels * (i + 1),
                          self.in_channels,
                          kernel_size=self.kernel_size,
                          dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1),
                    nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out  # type:ignore


# two lstm layers
class neck_block(nn.Module):
    """
    two lstm layers for neckblock
    para:
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bias,
                 batch_first,
                 dropout=0.1):
        super(neck_block, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.lstm1 = nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
        )
        self.lstm2 = nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
        )

    def forward(self, inputs: Tensor):
        x = self.lstm1(inputs)
        x = self.lstm2(inputs)
        return x


# dense+gru+dense
class stu_model(nn.Module):
    """
    stu_model:
        dense
        gru*2
        dense
    input:
        stft
        mfcc
    param:
        inchannels
        outchannels
        dilation
        input_size
        hidden_size
        bias
        dropout
        batchfirst
    """

    def __init__(self,
                 depth,
                 in_channels,
                 input_size,
                 hidden_size,
                 num_layers,
                 bias,
                 batch_first,
                 dropout=0.1):
        super(stu_model, self).__init__()
        self.encoder = DenseBlock(depth, in_channels)
        self.neck = neck_block(input_size,
                               hidden_size,
                               num_layers,
                               bias,
                               batch_first,
                               dropout=dropout)
        self.decoder = DenseBlock(depth, in_channels)

    def forward(self, input):
        x = self.encoder(input)
        print(x.size())
        B, C, F, T = x.size()
        x = x.reshape(B, -1, T)
        print(x.size())
        x, hn = self.neck(x)
        x = x.reshape(B, C, F, T)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    stumodel = stu_model(depth=4,
                         in_channels=2,
                         input_size=256,
                         hidden_size=256,
                         num_layers=2,
                         bias=True,
                         batch_first=True,
                         dropout=0.1)
    intensor = torch.rand(32, 2, 14, 256)
    output = stumodel(intensor)
    print(output.size())
