"""The alexnet backbone implementation.
"""
import logging

import torch.nn as nn

from utils.checkpoint import load_checkpoint


class AlexNet_Backbone(nn.Module):
    """The alexnet backbone.

    Args:
        nn.Module: The super class of base alexnet backbone.
    """

    def __init__(self, in_channel=3, lrn_param=[5, 1e-4, 0.75, 1.0], pretrained=None, init_weight=True, log_manager=None):
        """The initalization.

        Args:
            in_channel (int, optional): The input channels. Defaults to 3.
            lrn_param (list[float], optional): The LRN parameter. Defaults to [5, 1e-4, 0.75, 1.0].
            pretrained (str, optional): The pretrained weight path or None. Defaults to None.
            init_weight (bool, optional): The initalization of the weights option. Defaults to True.
            log_manager (builds.log.LogManager): The log manager. Defaults to None.
        """
        super(AlexNet_Backbone, self).__init__()

        #Setting Param
        self.in_channel = in_channel
        self.lrn_param = lrn_param
        self.pretrained = pretrained
        self.log_manager = log_manager

        self.conv1 = nn.Conv2d(self.in_channel, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1)

        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(*lrn_param)
        self.max_pooling = nn.MaxPool2d(3, 2)

        self.init_weight = init_weight

        if self.init_weight:
            self.init_weights()

    def forward(self, x):
        """The operation for every call.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The output features.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.max_pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.max_pooling(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        return x

    def init_weights(self):
        """The operation for initalization weights.
        
        Raises:
            TypeError : If pretrained type not in (None, str).
        """
        if isinstance(self.pretrained, str):
            if self.log_manager is not None:
                self.log_manager.logger.info(
                    'load pretrained model from {} to initalize the weights'.format(self.pretrained))

            load_checkpoint(self, self.pretrained)

        elif self.pretrained is None:
            if self.log_manager is not None:
                self.log_manager.logger.info('Initalize the weights')

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)

            nn.init.constant_(self.conv1.bias, 0)
            nn.init.constant_(self.conv2.bias, 1)
            nn.init.constant_(self.conv3.bias, 0)
            nn.init.constant_(self.conv4.bias, 1)
            nn.init.constant_(self.conv5.bias, 1)
        else:
            raise TypeError('The pretrained type must be in (None, str).')
