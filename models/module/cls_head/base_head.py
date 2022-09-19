"""Implement the base head class.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.metrics import *
import torch.nn as nn


from builds.loss import build_loss
from utils.convert import cvt2cat


class Base_Head(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, num_class, loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0), topk=(1, 5)):
        self.in_channels = in_channels
        self.num_class = num_class
        self.loss_cls = build_loss(loss_cls)

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
        for tk in topk:
            assert tk > 0, 'Topk should be larger than 0'
        self.topk = topk

    def loss(self, cls_scores, labels, **kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The ground truth.
        """
        #cls_socre : [[...,K],...,B]
        #soft label : [[...,K],...,B] => cls_score.size() == soft_label.size()
        #hard label : [(0~K-1),...,B] => cls_socre.size() != hard_label.size()
        losses = dict()

        if 'evaluation' in kwargs and kwargs['evaluation'] is not None:
            assert cls_scores.size() != labels.size(
            ), "We only support hard label when train/validation with evaluation"
            cls_scores_np = cvt2cat(cls_scores.detach().cpu().numpy())
            labels_np = labels.detach().cpu().numpy()

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
