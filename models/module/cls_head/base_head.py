"""Implement the base head class.
"""
from abc import ABCMeta, abstractmethod
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from builds.build import build
from utils.convert import cvt2cat, cvt2sps
from utils.check import check_cls


class Base_Head(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, num_class, loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0), label_smoothing_eps=0.0, multi_label=False):
        self.in_channels = in_channels

        assert num_class >= 2, "Number of the class must more than 2."
        self.num_class = num_class

        self.loss_cls = build(loss_cls)
        self.label_smoothing_eps = label_smoothing_eps
        self.multi_label = multi_label

    def loss(self, cls_scores, labels, **kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The ground truth.

        Binary class classification with BCE loss, N 2 + false
        cls_scores : B[prob]
        labels : B[prob|int]

        Multi class classification with CE loss, N 2초과 + false
        cls_scores : BxN[prob]
        labels : BxN[prob|int] | B[int]

        Multi label classification with BCE loss, N 2이상 + true
        cls_scores : BxN[prob]
        labels : BxN[prob|int]

        N : num of class(>=2), B : batch size, R : random
        """

        check_cls(cls_scores, labels, self.num_calss, self.multi_label)

        if self.label_smoothing_eps > 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, self.num_class)

            if self.multi_label:
                label_num = 2
            else:
                label_num = self.num_calss

            labels = ((1-self.label_smoothing_eps)*labels +
                      self.label_smoothing_eps/label_num)

        losses = dict()

        loss_cls = self.loss_cls(
            cls_scores, labels, **kwargs['loss']['loss_cls'] if 'loss_cls' in kwargs['loss'] else None)

        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        if kwargs['evaluation']:
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
        """Initialize the weights.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Define operation for every call.

        Args:
            x (torch.Tensor): The Features.
        """
        pass
