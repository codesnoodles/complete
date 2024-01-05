#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : resnet18.py
@Time    : 2023/12/26 22:32:47
@Author  : vhii
@Contact : zhangsworld@163.com
@Version : 0.1
'''
"""
教师网络实验
"""
import torch
import torchvision
import torchvision.transforms as transforms
import argparse, os
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
# img trans
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# down loda data
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=False,
                                        transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)


def train(args, net, device, train_loader, optimizer, epoch, scheduler):
    running_loss = 0.0
    correct = 0.
    batch_num = 0
    net.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        batch_num += 1
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('train epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, (batch_idx + 1) * len(inputs),
                len(train_loader.dataset), 100 * (batch_idx + 1) *
                len(inputs) / len(train_loader.dataset),
                running_loss / (batch_idx + 1)))
    scheduler.step()


def test(args, net, device, test_loader, train_loader, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='pytorch mnist example')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default 64)')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for testing (default 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=6,
                        metavar='N',
                        help='number of epochs to train (default 6)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        metavar='LR',
                        help='learning rate (default 0.001)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='sgd momentum(default 0.9)')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        help='disable cuda training ')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--save_model',
                        action='store_true',
                        default=True,
                        help='how many batches to wait before logging')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              **kwargs)
    net = resnet18(pretrained=True)
    net.load_state_dict(torch.load("resnet18-5c106cde.pth"))
    inchannel = net.fc.in_features
    net = net.to(device)
    print("create net", net)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)
    for epoch in range(args.epochs):
        train(args, net, device, train_loader, optimizer, epoch, scheduler)
        test(args, net, device, test_loader, train_loader, epoch)
    if (args.save_model):
        torch.save(net.state_dict(), "./cnn_resnet18.pth")


if __name__ == "__main__":
    main()
