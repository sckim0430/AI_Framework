"""The base class of classification implementation.
"""
from multiprocessing.sharedctypes import Value
import torch.nn as nn
from abc import ABCMeta, abstractclassmethod


class BaseClassification(nn.Module, metaclass=ABCMeta):
    """The base classification class.

    Args:
        torch.nn.Module: The torch class for definition of classification model.
    """

    def __init__(self, backbone, neck=None, cls_head=None, logger=None):
        """The initalization.

        Args:
            backbone (torch.nn.Module): The feature extractor.
            neck (torch.nn.Module): The feature refinementor.
            cls_head (torch.nn.Module, optional): The classification head. Defaults to None.
            logger (logging.RootLogger): The logger. Defaults to None.
        """
        super(BaseClassification, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.cls_head = cls_head
        self.logger = logger

    @abstractclassmethod
    def _forward_train(self, imgs, labels, **kwargs):
        """The train method.

        Args:
            imgs (torch.Tensor): The input images.
            labels (torch.Tensor): The input labels.
        """
        pass

    @abstractclassmethod
    def _forward_test(self, imgs):
        """The test method.

        Args:
            imgs (torch.Tensor): The input images.
        """
        pass

    def forward(self, imgs, labels=None, return_loss=True, **kwargs):
        """Define the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input images.
            labels (torch.Tensor, optional): The input labels.. Defaults to None.
            return_loss (bool, optional): The option for train/test. Defaults to True.

        Raises:
            ValueError: When return loss, the input labels should not be None.

        Returns:
            dict | torch.Tensor: The results of train or test.
        """
        if return_loss:
            if labels is None:
                raise ValueError('Label should not be None.')

            return self._forward_train(imgs, labels, **kwargs)
        else:
            return self._forward_test(imgs)

    def with_cls_head(self):
        return hasattr(self, "cls_head") and self.cls_head is not None

    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None
