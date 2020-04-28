import numpy as np
import torch as th

__all__ = [
    'ToChannelsFirst',
    'CastImages',
    'NormalizeImages',
    'ToTensor'
]


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
            'saliency': (sample['saliency'] - sample['saliency'].min()) / sample['saliency'].max(),
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
