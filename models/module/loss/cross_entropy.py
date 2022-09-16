"""Implement cross entropy loss.
"""
from base import BaseWeightedLoss
import torch
import torch.nn.functional as F

class CrossEntropyLoss(BaseWeightedLoss):
    """The cross entropy loss class.

    Args:
        BaseWeightedLoss (base.BaseWeigtedLoss): The super class with weighted loss.
    """
    def __init__(self,loss_weight=1.0,class_weight=None):
        """The initialization.

        Args:
            loss_weight (float, optional): The weight of loss. Defaults to 1.0.
            class_weight (_type_, optional): The weight of class. Defaults to None.
        """
        super().__init__(loss_weight=loss_weight)

        self.class_weight = class_weight

        if self.class_weight is not None:
            self.class_weight = torch.Tensor(self.class_weight)

    def _forward(self,cls_score,label,**kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated cross entropy loss.
        """

        if cls_score.size() == label.size():
            #Calculate with soft label
            assert cls_score.dim() == 2, "Only support 2 dimension soft label"
            assert len(kwargs)==0, "For now, no extra arguments are supproted for soft label"

            F.log_softmax(cls_score,1)
        else:
            #Calculate with hard label
            pass

        pass