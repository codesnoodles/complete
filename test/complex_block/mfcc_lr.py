#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : mfcc_lr.py
@Time    : 2023/12/26 15:53:08
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''

import librosa as lib
import numpy as np

data, sr = lib.load('./p257_431.wav')
print(data)
com_data = np.fft.fft(data)
print(com_data)
mag = np.abs(com_data)
print(mag)
s_data = lib.stft(data,
                  n_fft=2048,
                  hop_length=None,
                  win_length=None,
                  window='hann',
                  center=True,
                  pad_mode='reflect')
s_abs = np.abs(s_data)
s_db = lib.power_to_db(s_abs)
print(data.shape)  # 时域的
print(s_abs.shape)  # 频域的
print(s_db.shape)  # 分贝图
mel_data = lib.feature.melspectrogram(y=data,
                                      sr=sr,
                                      n_fft=2048,
                                      hop_length=512,
                                      n_mels=26)
mel_db = lib.power_to_db(mel_data)
print("mel_data:", mel_data.shape)
print("mel_db:", mel_db.shape)
