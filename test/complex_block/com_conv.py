#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : com_conv.py
@Time    : 2023/12/19 09:51:43
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''
"""
包含了基本conv和其他组件的conv模块
"""
import torch.nn as nn
import torch


class com_basicconv(nn.Module):
    """
     in_channels, 输入进来的通道数
     out_channels, 输出的通道数
     kernel_size, 使用的卷积核的大小
     stride=(1, 1), 卷积核移动的步长
     padding=(0, 0), 在边缘进行的拓展长度
     dilation=0, 空洞的大小
     groups=1, 分组卷积的大小
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
