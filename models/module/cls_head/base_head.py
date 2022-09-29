"""The base head Implementation.
"""
from abc import ABCMeta, abstractmethod
from sklearn.metrics import *
import torch
import torch.nn as nn

from builds.build import build
from utils.convert import cvt2sps
from utils.check import check_cls


class Base_Head(nn.Module, metaclass=ABCMeta):
    """The base classification head.

    Args:
        nn.Module: The super class of base classification head.
        metaclass (ABCMeta, optional): The abstract class. Defaults to ABCMeta.
    """

    def __init__(self, in_channels, num_class, loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0), multi_label=False, log_manager=None):
        """The initalization.

        Args:
            in_channels (int): The input channels.
            num_class (int): The number of class.
            loss_cls (dict, optional): The classification loss parameter. Defaults to dict(type='CrossEntropyLoss', loss_weight=1.0).
            multi_label (bool, optional): The multi label option. Defaults to False.
            log_manager (builds.log.LogManager): The log manager. Defaults to None.
        """
        self.in_channels = in_channels

        assert num_class >= 2, "Number of the class must more than 2."
        self.num_class = num_class

        self.loss_cls = build(loss_cls)
        self.multi_label = multi_label
        self.log_manager = log_manager

    def loss(self, cls_scores, labels, **kwargs):
        """The loss operation.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The ground truth.

        Binary class classification with BCE loss, N 2 + false
        cls_scores : B[float]
        labels : B[float|int]

        Multi class classification with CE loss, N 2초과 + false
        cls_scores : BxN[float]
        labels : BxN[float|int] | B[int]

        Multi label classification with BCE loss, N 2이상 + true
        cls_scores : BxN[float]
        labels : BxN[float|int]

        N : num of class(>=2), B : batch size, R : random
        """
        check_cls(cls_scores, labels, self.num_calss, self.multi_label)

        losses = dict()

        loss_cls = self.loss_cls(
            cls_scores, labels, **kwargs['loss']['loss_cls'] if 'loss_cls' in kwargs['loss'] else None)

        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        if kwargs['evaluation'] is not None:
            cls_scores_np = cls_scores.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            if self.multi_label or self.num_class == 2:
                labels_np = labels_np >= 0.5
                cls_scores_np = cls_scores_np >= 0.5
            else:
                if cls_scores_np.shape == labels_np.shape:
                    labels_np = cvt2sps(labels_np)
                cls_scores_np = cvt2sps(cls_scores_np)

            for func_name in kwargs['evaluation']:
                losses[func_name] = torch.tensor(eval(func_name)(labels_np, cls_scores_np,
                                                                 **kwargs['evaluation'][func_name]), device=cls_scores.device)

        return losses

    @abstractmethod
    def init_weights(self):
        """The weight initalization.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """The operation for every call.

        Args:
            x (torch.Tensor): The Features.
        """
        pass
