"""Implement cross entropy loss.
"""
from models.module.loss.base_loss import BaseWeightedLoss
import torch
import torch.nn.functional as F

class CrossEntropyLoss(BaseWeightedLoss):
    """The cross entropy loss class.

    Args:
        BaseWeightedLoss (base.BaseWeigtedLoss): The super class with weighted loss.
    """
    def __init__(self,loss_weight=1.0):
        """The initialization.

        Args:
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
            class_weight (list[float], optional): The weight of class. Defaults to None.
        """
        super().__init__(loss_weight=loss_weight)

    def _forward(self,cls_scores,labels,**kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated cross entropy loss.
        """

        if "weight" in kwargs and kwargs['weight']:
            kwargs['weight'] = torch.tensor(kwargs['weight'],device=cls_scores.device)
            
        loss_cls = F.cross_entropy(cls_scores,labels,**kwargs)

        return loss_cls