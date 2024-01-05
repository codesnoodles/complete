import torch.nn as nn


# simplenet
class simpleNet(nn.Module):

    def __init__(self, num_classes, input_nc):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(7),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * 2 * 64, 10)
        self.out = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 2 * 2 * 64)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.out(x)
        return x
