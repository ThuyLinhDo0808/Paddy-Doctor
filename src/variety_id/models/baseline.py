import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(Baseline, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (B, 256, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
