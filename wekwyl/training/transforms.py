import numpy as np
import torch as th


def _scale_values(x):
    diff = x.max() - x.min()
    if np.isclose(diff, 0):
        scaled = x.copy()
    else:
        scaled = (x - x.min()) / diff
    return scaled


class NoneIfEmpty:

    def __call__(self, sample):
        if (
            np.isclose(sample['frame'].max() - sample['frame'].min(), 0)
            or np.isclose(sample['saliency'].max() - sample['saliency'].min(), 0)
            or len(sample['fixations']) == 0
        ):
            return None

        return sample


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
            'saliency': _scale_values(sample['saliency']),
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
