"""AlexNet Class
"""
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet Architecture                                                                   

    Args:
        backbone (nn.Module) : alexnet backbone model class
        cls_head (nn.Module) : alexnet classification head model class
        init_weight (bool) : option for weight initialization
    """

    def __init__(self, backbone, cls_head):
        super(AlexNet, self).__init__()

        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.cls_head(x)
        loss = self.cls_head.loss()
        return x, loss_cls
