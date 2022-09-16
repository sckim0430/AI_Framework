"""Implement the base head class.
"""
from abc import ABCMeta, abstractmethod
import torch.nn as nn

from builds.loss import build_loss


class Base_Head(nn.Module, meta_class=ABCMeta):
    def __init__(self, in_channel, num_class, loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0), topk=(1, 5)):
        self.in_channel = in_channel
        self.num_class = num_class
        self.loss_cls = build_loss(loss_cls)

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
        for tk in topk:
            assert tk > 0, 'Topk should be larger than 0'
        self.topk = topk

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The ground truth.
        """
        losses = dict()
        
        # if labels.shape


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
