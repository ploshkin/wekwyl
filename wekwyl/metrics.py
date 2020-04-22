import math

import numpy as np
import torch as th
from torch import nn


class SphericalMSE(nn.Module):

    def __init__(self, h, w):
        super(SphericalMSE, self).__init__()
        self.h, self.w = h, w
        weight = th.sin(th.linspace(0.5, h - 0.5, steps=h) * math.pi / h)
        self.weight = nn.Parameter(weight.reshape((1, 1, h, 1)), requires_grad=False)

    def forward(self, y_pred, y_true):
        return th.mean((y_pred - y_true) ** 2 * self.weight)


class SphericalCC(nn.Module):

    def __init__(self, h, w):
        super(SphericalCC, self).__init__()
        self.h, self.w = h, w
        weight = th.sin(th.linspace(0.5, h - 0.5, steps=h) * math.pi / h)
        self.weight = nn.Parameter(weight.reshape((1, 1, h, 1)), requires_grad=False)

    def forward(self, y_pred, y_true):
        x = y_pred
        y = y_true
        vx = (x - th.mean(x, dim=[1, 2, 3]).reshape((-1, 1, 1, 1))) * self.weight
        vy = (y - th.mean(y, dim=[1, 2, 3]).reshape((-1, 1, 1, 1))) * self.weight
        return th.mean(
            th.sum(vx * vy, dim=[1, 2, 3])
            / (
                th.sqrt(th.sum(vx ** 2, dim=[1, 2, 3]))
                * th.sqrt(th.sum(vy ** 2, dim=[1, 2, 3]))
            )
        )


class SphericalNSS(nn.Module):

    def __init__(self, h, w):
        super(SphericalNSS, self).__init__()
        self.h, self.w = h, w
        kernels = self._compute_kernels(h)
        self.kernels = [
            nn.Parameter(kernel, requires_grad=False) for kernel in kernels
        ]

    def _compute_kernels(self, h):
        thetas = np.linspace(0.5, h - 0.5, num=h) * math.pi / h
        weight = 1 / np.sin(thetas)
        residual = weight % 2
        mask = residual >= 1
        residual[mask] -= 1
        residual[~mask] += 1
        n_ones = (weight - residual).astype(np.int32)
        edge_values = th.from_numpy(((weight - n_ones) / 2).astype(np.float32))
        kernels = [
            th.ones(size + 2) for size in n_ones
        ]
        for kernel, edge_value in zip(kernels, edge_values):
            kernel[..., [0, -1]] = edge_value

        return kernels

    def forward(self, y_pred, y_gt, eps=1e-5):
        assert y_pred.shape[0] == len(y_gt)
        assert y_pred.shape[1] == 1

        batch_size = y_pred.shape[0]

        num_fixations = th.Tensor(list(map(len, y_gt)))
        num_fixations[num_fixations < eps] = eps

        fixation_map = th.zeros_like(y_pred)
        for index, fixations in enumerate(y_gt):
            xs = np.rint(fixations[:, 0] * (self.w - 1)).astype(np.int32)
            ys = np.rint(fixations[:, 1] * (self.h - 1)).astype(np.int32)

            for x, y in zip(xs, ys):
                if y == 0 or y == self.h - 1:
                    fixation_map[index, :, y] = 1
                else:
                    kernel = self.kernels[y]
                    ker_w = kernel.shape[0]
                    left = x - ker_w // 2
                    right = left + ker_w

                    if left < 0:
                        length = -left
                        fixation_map[index, 0, y, (-left % self.w): ] = kernel[: length]
                        fixation_map[index, 0, y, : right] = kernel[length: ]

                    elif right >= self.w:
                        length = self.w - left
                        fixation_map[index, 0, y, left: ] = kernel[: length]
                        fixation_map[index, 0, y, : (right % self.w)] = kernel[length: ]

                    else:
                        fixation_map[index, 0, y, left: right] = kernel

        fixation_map = fixation_map.to(y_pred.device)
        return th.mean(th.sum(y_pred * fixation_map, dim=[1, 2, 3]) / num_fixations)
