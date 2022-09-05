import torch.nn as nn

class VGG11(nn.Module):
    def __init__(self,num_classes=1000,batch_norm=False,init_weights=True):
        #super(type,obj)
        #부모 클래스에게 자식 클래스의 type과 obj를 전달하고
        #부모 클래스의 생성자(__init__) 호출하여 자식 클래스의 멤버 변수 초기화(여기서는 초기화 없음)
        super(VGG11,self).__init__()

        if batch_norm:
            cnn_module = [] 

        self.features = nn.Sequential(
            nn.conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplcae=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.conv2d(64,128,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.conv2d(128,256,kernel_size=3,padding=1),
            nn.conv2d(256,256,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.conv2d(256,512,kernel_size=3,padding=1),
            nn.conv2d(512,512,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.conv2d(512,512,kernel_size=3,padding=1),
            nn.conv2d(512,512,kernel_size=3,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2)       
        )
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)