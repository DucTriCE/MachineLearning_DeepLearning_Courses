import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #1d = 3 dimens, 2d = 4 dimens (8, 3, 32, 32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #1d = 3 dimens, 2d = 4 dimens (8, 3, 32, 32)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_classes),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.create_module(3, 32) #output 8x32x112x112
        self.conv2 = self.create_module(32, 64) #output 8x64x56x56
        self.conv3 = self.create_module(64, 64) #output 8x64x28x28
        self.conv4 = self.create_module(64, 64) #output 8x64x14x14
        self.conv5 = self.create_module(64, 128) #output 8x128x7x7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=6272, out_features=1024),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
        )
    def create_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=out_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 1d = 3 dimens, 2d = 4 dimens (8, 3, 32, 32)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    images = torch.rand(8,3,224,224)
    model = AdvancedCNN(10)
    output = model(images)
    print(output.shape)
    print(output)