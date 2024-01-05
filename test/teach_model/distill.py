#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : distill.py
@Time    : 2023/12/26 22:31:12
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''

import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optime

from resnet18 import trainset, testset
from simplenet import simpleNet
from torchvision.models.resnet import resnet18

criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()


def train(teach_model, model, device, train_loader, optimizer, epoch,
          scheduler):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    alpha = 0.95
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss1 = criterion(outputs, labels)
        teacher_outputs = teach_model(inputs.float())
        T = 20
        outputs_S = F.log_softmax(outputs / T, dim=1)
        outputs_T = F.softmax(teacher_outputs / T, dim=1)
        loss2 = criterion2(outputs_S, outputs_T) * T * T
        loss = loss1 * (1 - alpha) + loss2 * alpha
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
        loss_sigma += loss.item()
        if i % 100 == 0:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print('loss_avg:{:.2}'.format(loss_avg))
            print("train epoch :{}[{}/{}({:0f}%)]\tLoss:{:.6f}".format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))
    scheduler.step()


def test(net, device, test_loader, train_loader, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, label, in test_loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set :Average loss:{:.4},accuracu:{}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    teach_model = resnet18(pretrained=True)  # different
    teach_model.load_state_dict(torch.load("resnet18-5c106cde.pth"))
    inchannel = teach_model.fc.in_features
    teach_model.fc = nn.Linear(inchannel, 10)
    teach_model = teach_model.to(device)
    model = simpleNet(10, 3).to(device)
    optimizer = optime.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)
    correct_ratio = []
    epoches = 100
    train_batches = 64
    test_batches = 64
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=train_batches,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batches,
                                              shuffle=False,
                                              **kwargs)
    for epoch in range(epoches):
        train(teach_model, model, device, train_loader, optimizer, epoch,
              scheduler)
        test(model, device, test_loader, train_loader, epoch)


if __name__ == "__main__":
    main()
