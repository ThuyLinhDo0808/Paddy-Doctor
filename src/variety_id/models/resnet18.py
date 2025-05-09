import torchvision.models as models
import torch.nn as nn


class RiceResnet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(RiceResnet18, self).__init__()

        self.model = models.resnet18(weights=None)

        # Replace the first conv layer if in_channels != 3 (e.g., for grayscale)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Replace the final FC layer to match num_classes
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 18),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(18, num_classes),
        )

    def forward(self, x):
        return self.model(x)
