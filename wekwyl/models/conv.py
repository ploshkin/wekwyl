import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _slice_channels(num_channels, num_pieces):
    assert num_channels >= num_pieces
    residual = num_channels % num_pieces
    piece_size = num_channels // num_pieces
    for index in range(num_pieces):
        if index < residual:
            yield piece_size + 1
        else:
            yield piece_size


def _pad_cylindric(input, pad):
    l, r, u, d = pad
    w = input.shape[3]

    up = input[:, :, 1: u + 1]
    up = th.roll(up, shifts=w // 2, dims=[3])
    up = th.flip(up, dims=[2])

    down = input[:, :, -(d + 1): -1]
    down = th.roll(down, shifts=w // 2, dims=[3])
    down = th.flip(down, dims=[2])

    padded = th.cat([up, input, down], dim=2)
    return F.pad(padded, [l, r, 0, 0], mode='circular')


class CylindricConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(CylindricConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.paddings = [
            (w // 2, w // 2, h // 2, h // 2)
            for h, w in kernel_sizes
        ]
    
        num_groups = len(kernel_sizes)
        self.in_groups = list(_slice_channels(in_channels, num_groups))
        self.out_groups = list(_slice_channels(out_channels, num_groups))
    
        self.convolutions = [
            nn.Conv2d(
                in_ch,
                out_ch,
                size,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            for in_ch, out_ch, size in zip(
                self.in_groups, self.out_groups, kernel_sizes,
            )
        ]

    def forward(self, x):
        groups = th.split(x, self.in_groups, dim=1)
        convolved = [
            conv(_pad_cylindric(group, pad))
            for conv, group, pad in zip(
                self.convolutions, groups, self.paddings,
            )
        ]
        return th.cat(convolved, dim=1)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for i in range(len(self.convolutions)):
            self.convolutions[i] = self.convolutions[i].to(*args, **kwargs)
        return self
