import torch
import torch.nn as nn


class CA_Cell(nn.Module):
    def __init__(self, channels):
        super(CA_Cell, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_features=channels, out_features=round(channels / 16), bias=False)
        self.fc2 = nn.Linear(in_features=round(channels / 16), out_features=channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z_mean = self.gap(x)
        z_max = self.max(x)
        z_mix = z_max + z_mean
        z_mix = z_mix.view(z_mix.size(0), -1)
        z = self.fc2(self.relu(self.fc1(z_mix)))
        return z.view(z.size(0), z.size(1), 1, 1)


class SA_Cell(nn.Module):
    def __init__(self):
        super(SA_Cell, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)
        mix = mean + max[0]
        return self.relu(mix)


class PAUnit(nn.Module):

    def __init__(self, channels):
        super(PAUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.ca = CA_Cell(channels)
        self.sa = SA_Cell()
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        k = self.conv2(self.relu(self.conv1(x)))
        v = k * x
        vc = self.ca(v)
        vs = self.sa(v)
        return torch.cat([self.gap(vc*x), self.gap(vs*x)], dim=1)

