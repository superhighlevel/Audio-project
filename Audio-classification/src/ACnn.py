import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.conv_block(x) 

class ACNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=1, out_channels=16,
            kernel_size=3, stride=1, padding=2,
            )
        self.conv_block2 = ConvBlock(
            in_channels=16, out_channels=32,
            kernel_size=3, stride=1, padding=2,
            )
        self.conv_block3 = ConvBlock(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=1, padding=2,
        )
        self.conv_block4 = ConvBlock(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=2,
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128*5*4, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input):
        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        logits = self.dense(x)
        preds = self.softmax(logits)
        return preds 

if __name__=='__main__':
    acnn = ACNNNetwork()
    summary(acnn.cuda(), (1, 64, 44))


