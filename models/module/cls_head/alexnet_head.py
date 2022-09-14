"""AlexNet Head Class
"""
import torch.nn as nn
from builds.loss import build_loss


class AlexNet_Head(nn.Module):
    """AlexNet Head Architecture

    Args:
        in_channel (int) : number of channel in input feature
        num_class (int) : number of class to classification
        dropout_ratio (float) : dropout ratio
        pooling_type (str) : option for global average pooling
        loss (dict) : loss option
    """

    def __init__(self, num_class=1000, in_channel=256, dropout_ratio=0.5, pooling_type="avg", loss=dict(type="CrossEntropy")):
        self.num_class = num_class
        self.in_channel = in_channel
        self.dropout_ratio = dropout_ratio
        self.pooling_type = pooling_type
        self.loss = build_loss(loss)

        self.fc1 = nn.Linear(6*6*self.in_channel, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        if self.pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(6)
        else:
            self.pooling = nn.MaxPool2d(3, 2)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.softmax = nn.Softmax(dim=1)

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
        x = self.softmax(x)

        # cls_loss = self.loss(x)
        # return x, cls_loss
