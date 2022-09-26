"""AlexNet Head Class
"""
import torch.nn as nn
from base_head import Base_Head


class AlexNet_Head(Base_Head):
    """AlexNet Head Architecture

    Args:
        Base_Head (base_head.Base_Head): The super class of the AlexNet head.
    """
    def __init__(self, num_class=1000, in_channels=256, dropout_ratio=0.5, pooling_type="avg", loss_cls=dict(type="CrossEntropy",loss_weight=1.0), multi_label=False, init_weight=True):
        """The initalization.

        Args:
            num_class (int, optional): The number of class to classification. Defaults to 1000.
            in_channel (int, optional): The number of channel in input feature. Defaults to 256.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
            pooling_type (str, optional): The option for global average pooling. Defaults to "avg".
            loss (dict, optional): The loss option. Defaults to dict(type="CrossEntropy").
            init_weight (bool, optional): The option for initalization of the weights. Defaults to True.
        """
        super().__init__(num_class,in_channels,loss_cls,multi_label)
        self.num_class = num_class
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.pooling_type = pooling_type

        self.fc1 = nn.Linear(6*6*self.in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_class)

        if self.pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(6)
        else:
            self.pooling = nn.MaxPool2d(3, 2)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.softmax = nn.Softmax(dim=1)

        self.init_weight = init_weight

        if self.init_weight:
            self.init_wiehgts()

    def forward(self, x):
        x = self.pooling(x)
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
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,1)