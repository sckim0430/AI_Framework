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

    def _forward(self,cls_scores,labels,**kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated cross entropy loss.
        """
        if cls_scores.size() == labels.size():
            #Calculate with soft label
            assert cls_scores.dim() == 2, "Only support 2 dimension soft label"
            
            


            F.log_softmax(cls_scores,1)
        else:
            #Calculate with hard label
            pass

        pass