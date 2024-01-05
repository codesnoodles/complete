#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : mfcc.py
@Time    : 2023/12/21 19:24:49
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''
"""
test of mfcc
"""
import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load('./p257_431.wav', sr=16000)
# 提取 mel spectrogram feature
melspec = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         n_fft=1024,
                                         hop_length=512,
                                         n_mels=128)
logmelspec = librosa.power_to_db(melspec)  # 转换为对数刻度
# 绘制 mel 频谱图
plt.figure()
librosa.display.specshow(
    logmelspec,
    sr=sr,
    x_axis='time',
    y_axis='mel',
)
plt.colorbar(format='%+2.0f dB')  # 右边的色度条
plt.title('Beat wavform')
plt.show()
