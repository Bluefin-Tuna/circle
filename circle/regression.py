from circle.utils import iou
from torch import nn

class ConvNet(nn.Module):
    
    def __init__(self) -> None:
        
        super(ConvNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(256, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU()
        )
        self.y = nn.Linear(16, 3)

    def forward(self, x):
        x = self.l5(self.l4(self.l3(self.l2(self.l1(x)))))
        _, C, H, W = x.shape
        x = x.view(-1, C * H * W)
        x = self.fc(x)
        x = self.y(x)
        return x