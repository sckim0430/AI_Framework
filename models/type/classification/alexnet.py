import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, backbone, cls_head, init_weight=True):
        super(AlexNet, self).__init__()

        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.cls_head = cls_head

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.cls_head(x)

    def _initialize_weights(self):
        for idx, m in enumerate(self.modules()):
            nn.init.normal(m.weight,mean=0.0,std=0.01)
            #bias conv 2,4,5 & fc layer 1
            if idx in []:
                pass
            else:
                pass
