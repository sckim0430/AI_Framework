"""The weighted loss Implementation.
"""

from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """The weighted loss.

    Args:
        nn.Module: The super class of weighted loss.
        metaclass (ABCMeta, optional): The abstract class. Defaults to ABCMeta.
    """

    def __init__(self, loss_weight=1.0):
        """The initalization.

        Args:
            loss_weight (float, optional): The loss weight. Defaults to 1.0.
        """
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        """The operation for every call.
        """
        pass

    def forward(self, *args, **kwargs):
        """The operation for every call.

        Returns:
            torch.Tensor: The loss.
        """
        ret = self._forward(*args, **kwargs)

        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            ret *= self.loss_weight

        return ret
