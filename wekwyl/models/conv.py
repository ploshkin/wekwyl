import torch as th
import torch.nn as nn

from .pad import pad_cylindric_2d


def slice_channels(num_channels, num_pieces):
    assert num_channels >= num_pieces
    residual = num_channels % num_pieces
    piece_size = num_channels // num_pieces
    for index in range(num_pieces):
        if index < residual:
            yield piece_size + 1
        else:
            yield piece_size


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
        self.in_groups = list(slice_channels(in_channels, num_groups))
        self.out_groups = list(slice_channels(out_channels, num_groups))
    
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
            conv(pad_cylindric_2d(group, pad))
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
