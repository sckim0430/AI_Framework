"""AlexNet Head Class
"""
import torch.nn as nn

class AlexNet_Head(nn.Module):
    """AlexNet Head Architecture

    Args:
        in_channel (int) : number of channel in input feature
        num_class (int) : number of class to classification
        dropout_ratio (float) : dropout ratio
        loss (dict) : loss option
    """
    def __init__(self,num_class=1000, dropout_ratio=0.5, loss=dict(type="CrossEntropy")):
        self.num_class = num_class
        self.dropout_ratio = dropout_ratio
        # self.loss = build_loss(loss)

        self.fc1 = nn.Linear(6*6*256,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        
        self.pooling = nn.MaxPool2d(3,2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
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
