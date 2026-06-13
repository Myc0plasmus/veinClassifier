import torch
import torch.nn as nn

class basicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__();
        self.net = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, 
                              out_channels=16,
                              kernel_size=5,
                              padding=2),
                    nn.ReLU(),
                    nn.Conv2d(16, out_channels, kernel_size=1)
                )
    def forward(self, x):
        return self.net(x)
