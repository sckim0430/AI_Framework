"""AlexNet Backbone Class
"""
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet Backbone Architecture

    Args:
        in_channel (int) : number of channel in input feature
        out_channel (int) : number of channel in out feature
        lrn_param (list) : local response norm parameters
        pretrained (bool) : option for using pretrained model weight
    """

    def __init__(self, in_channel=3, lrn_param=[5, 1e-4, 0.75, 1.0], pretrained=False):
        super(AlexNet, self).__init__()

        #Setting Param
        self.in_channel = in_channel
        self.lrn_param = lrn_param
        self.pretrained = pretrained

        self.conv1 = nn.Conv2d(self.in_channel, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1)

        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(*lrn_param)
        self.max_pooling = nn.MaxPool2d(3, 2)

    def forward(self, x):
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
