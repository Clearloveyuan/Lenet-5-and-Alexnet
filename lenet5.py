import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 32, 32)
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0
            ),  # ->(16, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)  # ->(16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),  # (16, 10, 10)
            nn.ReLU(),
            nn.AvgPool2d(2)              # (16, 5, 5)
        )
        self.out = nn.Sequential(
            nn.Linear(16 * 5 * 5 , 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将输出化成一维向量
        output = self.out(x)
        return output

myNet = LeNet5()
print(myNet)