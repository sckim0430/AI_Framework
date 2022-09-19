"""Implement super class with weighted loss.
"""

from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """The super class with weighted loss.

    Args:
        torch.nn.Module : The torch class for loss definition.
        meta_class (abc.ABCMeta, optional): The abstract base class. Defaults to ABCMeta.
    """

    def __init__(self, loss_weight=1.0):
        """The Initialization.

        Args:
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
        """
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        """The abstract method to calculate the loss.
        """
        pass

    def forward(self, *args, **kwargs):
        """Calculate the loss.

        Returns:
            torch.Tensor: The calculated loss.
        """
        ret = self._forward(*args,**kwargs)

        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
                ret *= self.loss_weight
        
        return ret