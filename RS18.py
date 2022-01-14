import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Starting channels
        self.num_in = 64
        # First convolution layer in ResNet18 generally has 7x7 kernel size,
        # but that was for images of 224*224, so I lowered it to kernel size 3.
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64))
        # 4 groups of blocks, each with 2 layers
        self.block1 = self.make_layers(ResNetBlock, num_in=64, stride=1)
        self.block2 = self.make_layers(ResNetBlock, num_in=128, stride=2)
        self.block3 = self.make_layers(ResNetBlock, num_in=256, stride=2)
        self.block4 = self.make_layers(ResNetBlock, num_in=512, stride=2)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Last fully conencted layer for 8 classes
        self.full = nn.Linear(512, 8)
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.avgpool(out)
        # Flatten
        out = out.view(out.size(0), -1)
        out = self.full(out)
        return out

    # Makes each of the 4 groups of multiple convolutional blocks
    def make_layers(self, block, num_in, stride):
        layers = []
        downsample = nn.Sequential(nn.Conv2d(self.num_in, num_in, kernel_size=1, stride=stride),
                                   nn.BatchNorm2d(num_in))
        layers.append(block(self.num_in, num_in, downsample, stride))
        self.num_in = num_in
        layers.append(block(self.num_in, num_in))
        return nn.Sequential(*layers)

class ResNetBlock(nn.Module):
    def __init__(self, num_in, num_out, downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.conv2 = nn.Conv2d(num_out, num_out, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, input):
        identity = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # May need extra downsample layer before adding it to residual block
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
