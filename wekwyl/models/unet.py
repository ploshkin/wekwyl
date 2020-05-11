import functools

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .conv import CylindricConv2d


class _UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
            self,
            n_out,
            n_flt,
            n_in=None,
            subnet=None,
            downsample=False,
            outer=False,
            inner=False,
            norm_layer=nn.BatchNorm2d,
            kernel_sizes=[(3, 3)],
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            n_out (int) -- the number of filters in the outer conv layer
            n_flt (int) -- the number of filters in the inner conv layer
            n_in (int) -- the number of channels in input images/features
            subnet (_UnetBlock) -- previously defined submodules
            downsample (bool) -- if downsample
            outer (bool) -- if this module is the outermost module
            inner (bool) -- if this module is the innermost module
            norm_layer -- normalization layer
            kernel_sizes (list) -- list of kernel sizes inside CylindricConv2d
        """
        super(_UnetBlock, self).__init__()
        self.outer = outer

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if not n_in:
            n_in = n_out

        if outer:
            layers = [
                CylindricConv2d(n_in, n_flt, kernel_sizes, stride=2, bias=use_bias),
                norm_layer(n_flt),
                subnet,
                nn.LeakyReLU(0.2, inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                CylindricConv2d(n_flt * 2, n_flt, kernel_sizes, bias=use_bias),
                norm_layer(n_flt),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_flt, n_out, (1, 1), bias=use_bias),
                norm_layer(n_out),
                nn.Sigmoid(),
            ]

        elif inner:
            layers = [
                nn.LeakyReLU(0.2, inplace=True),
                CylindricConv2d(n_in, n_flt, kernel_sizes, stride=2, bias=use_bias),
                norm_layer(n_flt),
                nn.ReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                CylindricConv2d(n_flt, n_out, kernel_sizes, bias=use_bias),
                norm_layer(n_out),
            ]

        else:
            layers = [
                nn.LeakyReLU(0.2, inplace=True),
                CylindricConv2d(n_in, n_flt, kernel_sizes, stride=2, bias=use_bias),
                norm_layer(n_flt),
                subnet,
                nn.ReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
                CylindricConv2d(n_flt * 2, n_out, kernel_sizes, bias=use_bias),
                norm_layer(n_out),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.outer:
            return self.model(x)
        else:   # Add skip connection.
            return th.cat([x, self.model(x)], dim=1)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        for module in self.model.modules():
            module = module.to(*args, **kwargs)
        return self


class CylindricUnet(nn.Module):

    """Create a Unet-based generator"""

    def __init__(
            self,
            n_in,
            n_out,
            n_downs,
            ngf,
            norm_layer=nn.BatchNorm2d,
            kernel_sizes=[(3, 3)],
    ):
        """Construct a Unet generator
        Parameters:
            n_in (int) -- the number of channels in input images
            n_out (int) -- the number of channels in output images
            n_downs (int) -- the number of downsamplings in UNet. For example, if |n_downs| == 7,
                             image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            kernel_sizes -- list of kernel sizes used in CylindricConv2d
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(CylindricUnet, self).__init__()
        # Add the innermost layer.
        subnet = _UnetBlock(
            ngf * 8,
            ngf * 8,
            norm_layer=norm_layer,
            inner=True,
            kernel_sizes=kernel_sizes,
        )
        # Add intermediate layers with ngf * 8 filters.
        for i in range(n_downs - 5):
            subnet = _UnetBlock(
                ngf * 8,
                ngf * 8,
                subnet=subnet,
                norm_layer=norm_layer,
                kernel_sizes=kernel_sizes,
            )

        # Gradually reduce the number of filters from ngf * 8 to ngf.
        subnet = _UnetBlock(
            ngf * 4,
            ngf * 8,
            subnet=subnet,
            norm_layer=norm_layer,
            kernel_sizes=kernel_sizes,
        )
        subnet = _UnetBlock(
            ngf * 2,
            ngf * 4,
            subnet=subnet,
            norm_layer=norm_layer,
            kernel_sizes=kernel_sizes,
        )
        subnet = _UnetBlock(
            ngf,
            ngf * 2,
            subnet=subnet,
            norm_layer=norm_layer,
            kernel_sizes=kernel_sizes,
        )
        # Add the outermost layer.
        self.model = _UnetBlock(
            n_out,
            ngf,
            n_in=n_in,
            subnet=subnet,
            outer=True,
            norm_layer=norm_layer,
            kernel_sizes=kernel_sizes,
        )

    def forward(self, x):
        x = self.model(x)
        prob = x / x.sum(dim=[1, 2, 3]).reshape((-1, 1, 1, 1))
        return {
            'prob': prob,
            'logit': th.log(prob),
        }

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        for module in self.model.modules():
            module = module.to(*args, **kwargs)
        return self
