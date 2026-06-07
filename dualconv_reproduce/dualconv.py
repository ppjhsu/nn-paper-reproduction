import torch
import torch.nn as nn


class DualConv(nn.Module):
    """DualConv = 3x3 group convolution + 1x1 pointwise convolution.

    Paper definition: the two branches process the same input feature map
    simultaneously and their outputs are summed.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, g: int = 4, bias: bool = False):
        super().__init__()
        if in_channels % g != 0 or out_channels % g != 0:
            # PyTorch grouped conv requires both in/out channels divisible by groups.
            # Fall back to the largest valid group count not exceeding g.
            valid = [x for x in range(min(in_channels, out_channels, g), 0, -1)
                     if in_channels % x == 0 and out_channels % x == 0]
            g = valid[0]
        self.g = g
        self.group_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, groups=g, bias=bias
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride,
            padding=0, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_conv(x) + self.pointwise_conv(x)
