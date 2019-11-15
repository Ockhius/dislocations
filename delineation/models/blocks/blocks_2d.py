import torch
import torch.nn as nn
import torch.nn.functional as F


class res_2d_block(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(res_2d_block,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out


class res_2d_block_in(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(res_2d_block_in,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.in1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.in2 = nn.InstanceNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out=F.relu(self.in1(self.conv1(x)))
        out=self.in2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out