"""The alexnet head Implementation.
"""
import torch.nn as nn
from .base_head import Base_Head


class AlexNet_Head(Base_Head):
    """The alexnet head.

    Args:
        Base_Head (base_head.Base_Head): The super class of the AlexNet head.
    """

    def __init__(self, num_class=1000, in_size=6, in_channel=256, loss_cls=dict(type="CrossEntropy", loss_weight=1.0), avg_pooling=True, multi_label=False, init_weight=True, dropout_ratio=0.5, logger=None):
        """The initalization.

        Args:
            num_class (int, optional): The number of class. Defaults to 1000.
            in_size (int|list[int], optional): The input size. Defaults to 6.
            in_channel (int, optional): The input channels. Defaults to 256.
            loss_cls (dict, optional): The classification loss parameter. Defaults to dict(type='CrossEntropyLoss', loss_weight=1.0).
            avg_pooling (bool, optional): The average pooling option for input featrue. Defaults to True.
            multi_label (bool, optional): The multi label option. Defaults to False.
            init_weight (bool, optional): The initalization of the weights option. Defaults to True.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
            logger (logging.RootLogger): The logger. Defaults to None.
        """
        super(AlexNet_Head, self).__init__(num_class=num_class, in_size=in_size, in_channel=in_channel,
                                           loss_cls=loss_cls, avg_pooling=avg_pooling, multi_label=multi_label, logger=logger)
        self.dropout_ratio = dropout_ratio

        self.fc1 = nn.Linear(
            self.in_height*self.in_width*self.in_channel, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_class)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio)

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
        if self.avg_pooling is not None:
            x = self.avg_pooling(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def init_weights(self):
        """The operation for initalization weights.
        """
        if self.logger is not None:
            self.logger.info('Initalize the weights of AlexNet head.')

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 1)
