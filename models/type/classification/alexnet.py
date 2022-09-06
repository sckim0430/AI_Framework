"""AlexNet Class
"""
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet Architecture

    Args:
        backbone (nn.Module) : AlexNet backbone model class
        cls_head (nn.Module) : AlexNet classification head model class
        init_weight (bool) : option for weight initialization
    """
    def __init__(self, backbone, cls_head, init_weight=True):
        super(AlexNet, self).__init__()

        self.backbone = backbone
        self.cls_head = cls_head

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x, loss_cls = self.cls_head(x)
        return x, loss_cls

    def _initialize_weights(self):
        for idx, m in enumerate(self.modules()):
            nn.init.normal(m.weight, mean=0.0, std=0.01)
            #bias conv 2,4,5 & fc layer 1
            if idx in []:
                pass
            else:
                pass
