import torch as th
import torch.nn.functional as F


def pad_cylindric_2d(input, pad):
    l, r, u, d = pad
    padded = F.pad(input, [l, r, 0, 0], mode='circular')
    return F.pad(padded, [0, 0, u, d], mode='constant')
