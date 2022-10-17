"""The cross entropy loss implementation.
"""
from models.module.loss.base_loss import BaseWeightedLoss
import torch
import torch.nn.functional as F


class CrossEntropyLoss(BaseWeightedLoss):
    """The cross entropy loss.

    Args:
        BaseWeightedLoss (base.BaseWeigtedLoss): The super class of the cross entropy.
    """

    def __init__(self, loss_weight=1.0):
        """The initalization.

        Args:
            loss_weight (float, optional): The loss weight. Defaults to 1.0.
        """
        super(CrossEntropyLoss,self).__init__(loss_weight=loss_weight)

    def _forward(self, cls_scores, labels, **kwargs):
        """The operation for every call.

        Args:
            cls_score (torch.Tensor): The class scores.
            label (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The cross entropy loss.
        """
        if "weight" in kwargs and kwargs['weight'] is not None:
            kwargs.update(dict(weight=torch.tensor(
                kwargs['weight'], device=cls_scores.device)))

        loss_cls = F.cross_entropy(cls_scores, labels, **kwargs)

        return loss_cls
