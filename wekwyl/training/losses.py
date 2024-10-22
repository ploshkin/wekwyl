import math

import numpy as np
import torch as th
from torch import nn


def _normalize_batch(x, eps=1e-5):
    mean = x.mean(dim=[1, 2, 3]).reshape((-1, 1, 1, 1))
    std = x.std(dim=[1, 2, 3]).reshape((-1, 1, 1, 1))
    res = th.zeros_like(std)
    res[std < eps] = eps
    return (x - mean) / (std + res)


class SphericalMSE(nn.Module):

    def __init__(self, h, w):
        super(SphericalMSE, self).__init__()
        self.h, self.w = h, w
        weight = th.sin(th.linspace(0.5, h - 0.5, steps=h) * math.pi / h)
        self.weight = nn.Parameter(weight.reshape((1, 1, h, 1)), requires_grad=False)

    def forward(self, y_pred, y_true):
        return th.mean((y_pred - y_true) ** 2 * self.weight)


class SphericalCC(nn.Module):

    def __init__(self, h, w, use_weight=True, eps=1e-5):
        super(SphericalCC, self).__init__()
        self.h, self.w = h, w
        self.use_weight = use_weight
        self.eps = eps
        weight = th.sin(th.linspace(0.5, h - 0.5, steps=h) * math.pi / h)
        self.weight = nn.Parameter(weight.reshape((1, 1, h, 1)), requires_grad=False)

    def _center(self, x):
        mean = x.mean(dim=[1, 2, 3]).reshape((-1, 1, 1, 1))
        return x - mean

    def _std(self, x):
        sx = th.sqrt(th.sum(x ** 2, dim=[1, 2, 3]))
        res = th.zeros_like(sx)
        res[sx < self.eps] = self.eps
        return sx + res

    def forward(self, y_pred, y_true):
        vy_pred = self._center(y_pred)
        vy_true = self._center(y_true)

        if self.use_weight:
            vy_pred *= self.weight
            vy_true *= self.weight

        sy_pred = self._std(vy_pred)
        sy_true = self._std(vy_true)

        return th.mean(
            th.sum(vy_pred * vy_true, dim=[1, 2, 3])
            / (sy_pred * sy_true)
        )


class SphericalNSS(nn.Module):

    def __init__(self, h, w, eps=1e-5):
        super(SphericalNSS, self).__init__()
        self.h, self.w = h, w
        self.eps = eps
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

    def forward(self, y_pred, y_gt):
        assert y_pred.shape[0] == len(y_gt)
        assert y_pred.shape[1] == 1

        batch_size = y_pred.shape[0]

        num_fixations = th.tensor(list(map(len, y_gt)), device=y_pred.device)
        num_fixations[num_fixations < self.eps] = self.eps

        fixation_map = th.zeros_like(y_pred)
        for index, fixations in enumerate(y_gt):
            fixations = np.array(fixations)
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
                        fixation_map[index, 0, y, (left % self.w): ] = kernel[: length]
                        fixation_map[index, 0, y, : right] = kernel[length: ]

                    elif right >= self.w:
                        length = self.w - left
                        fixation_map[index, 0, y, left: ] = kernel[: length]
                        fixation_map[index, 0, y, : (right % self.w)] = kernel[length: ]

                    else:
                        fixation_map[index, 0, y, left: right] = kernel

        y = _normalize_batch(y_pred, self.eps)
        return th.mean(th.sum(y * fixation_map, dim=[1, 2, 3]) / num_fixations)
