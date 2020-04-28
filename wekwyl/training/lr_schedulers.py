import math

from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    'CosineAnnealingWithWarmupLR'
]


class CosineAnnealingWithWarmupLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with linear warmup at the beginning.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        warmup_epochs (int): Number of iterations with linear scaling of the learning rate.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)

    def _linear_coeff(self):
        return self.last_epoch / self.warmup_epochs

    def _cosine_coeff(self):
        max_epochs = self.T_max - self.warmup_epochs
        num_epochs = self.last_epoch - self.warmup_epochs
        return (1 + math.cos(math.pi * num_epochs / max_epochs)) / 2

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            k = self._linear_coeff()
        else:
            k = self._cosine_coeff()
        return [
            self.eta_min + k * (base_lr - self.eta_min)
            for base_lr in self.base_lrs
        ]
