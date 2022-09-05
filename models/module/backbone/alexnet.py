import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,lrn_param=[5,1e-4,0.75,1.0],pretrained=False):
        super(AlexNet,self).__init__()
        self.conv1 =  nn.Conv2d(3,96,11,4)
        self.conv2 = nn.Conv2d(96,256,5)
        self.conv3 = nn.Conv2d(256,384,3)
        self.conv4 = nn.Conv2d(384,384,3)
        self.conv5 = nn.Conv2d(384,256,3)
        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(*lrn_param)
        self.max_pooling = nn.MaxPool2d(3,2)

    def forward(self,x):
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
        x = self.max_pooling(x)

        return x