#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : complexnet.py
@Time    : 2023/12/14 22:00:02
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''

import torch
import torch.nn as nn
from .utils import *
from torchinfo import summary
from .encoder import *
from .decoder import *
from .neckblock import *
from .conv_modules import *


class Model_dpt(nn.Module):

    def __init__(self, num_channel=64, num_features=257):
        super(Model_dpt, self).__init__()
        self.encoder = EncoderBlock_dpt(in_channels=2, out_channels=64)
        self.neck = ConnectTrans(input_size=64, output_size=64)
        self.conv1 = ConvGate(in_channels=64,
                              out_channels=64,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              dilation=(1, 1))
        self.decoder = ComplexDecoder_dpt()

    def forward(self, x):
        en_out = self.encoder(x)
        neck_out = self.neck(en_out)
        conv_out = self.conv1(neck_out)
        de_out = self.decoder(conv_out)
        out = x * de_out
        final_real = out[:, 0, :, :].unsqueeze(1)
        final_imag = out[:, 1, :, :].unsqueeze(1)
        return final_real, final_imag


# 模型组件
# class BasicConv(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=(1, 1),
#                  padding=(0, 0),
#                  dilation=1,
#                  groups=1,
#                  relu=True,
#                  inorm=True,
#                  bias=False):
#         super().__init__()

#         self.conv = nn.Conv2d(in_channels,
#                               out_channels,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               dilation=dilation,
#                               groups=groups,
#                               bias=bias)
#         self.inorm = nn.InstanceNorm2d(out_channels,
#                                        affine=True) if inorm else None
#         self.relu = nn.PReLU() if relu else None

#     def forward(self, x):

#         x = self.conv(x)
#         if self.inorm is not None:
#             x = self.inorm(x)
#         if self.relu is not None:
#             x = self.relu(x)


#         return x
class com_basicconv(nn.Module):
    """
     in_channels, 输入进来的通道数
     out_channels, 输出的通道数
     kernel_size, 使用的卷积核的大小
     stride=(1, 1), 卷积核移动的步长
     padding=(0, 0), 在边缘进行的拓展长度
     dilation=0, 空洞的大小
     groups=1, 分组卷积的大小
     relu=True, 是否使用激活函数
     inorm=True, 是否包含norm层
     bias=False 卷积的时候是否包含偏置
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=0,
                 groups=1,
                 relu=True,
                 inorm=True,
                 bias=False):
        super().__init__()
        self.conv_i = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )  # 处理虚部的卷积层
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )  #处理实部的卷积层
