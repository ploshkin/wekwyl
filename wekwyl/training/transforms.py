import numpy as np
import torch as th


def _scale_values(x):
    x_min = x.min()
    d = x.max() - x_min
    if np.isclose(d, 0):
        return x - x_min
    else:
        return (x - x_min) / d


def _to_pdf(x):
    x = x - x.min()
    s = x.sum()
    if np.isclose(s, 0):
        return th.full(x.shape, 1 / x.numel())
    else:
        return x / s


class ToChannelsFirst:

    def __call__(self, sample):
        return {
            'frame': sample['frame'].transpose((2, 0, 1)),
            'saliency': sample['saliency'].transpose((2, 0, 1)),
            'fixations': sample['fixations'],
        }


class CastImages:
    def __init__(self, numpy_type):
        self.numpy_type = numpy_type

    def __call__(self, sample):
        return {
            'frame': sample['frame'].astype(self.numpy_type),
            'saliency': sample['saliency'].astype(self.numpy_type),
            'fixations': sample['fixations'],
        }


class NormalizeImages:

    def __call__(self, sample):
        return {
            'frame': sample['frame'] / 255.,
            'saliency': _to_pdf(sample['saliency']),
            'fixations': sample['fixations'],
        }


class ToTensor:

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return {
            'frame': th.from_numpy(sample['frame']).to(self.device),
            'saliency': th.from_numpy(sample['saliency']).to(self.device),
            'fixations': sample['fixations'],
        }
