from torch import nn
import torch
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_feats = self.max_pool(x)
        avg_feats = self.avg_pool(x)

        max_feats = torch.flatten(max_feats, 1)
        avg_feats = torch.flatten(avg_feats, 1)

        max_feats = self.mlp(max_feats)
        avg_feats = self.mlp(avg_feats)

        output = (
            self.sigmoid(max_feats + avg_feats).unsqueeze(2).unsqueeze(3).expand_as(x)
        )

        return output * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)

        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()

        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out


class CBAMResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CBAMResNet18, self).__init__()

        # Load vanilla ResNet18
        base = models.resnet18(weights=None)

        # Replace first conv layer if needed
        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # --- Copy stem and stages ---
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        self.layer1 = base.layer1
        self.cbam1 = CBAM(64)

        self.layer2 = base.layer2
        self.cbam2 = CBAM(128)

        self.layer3 = base.layer3
        self.cbam3 = CBAM(256)

        self.layer4 = base.layer4
        self.cbam4 = CBAM(512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.pool(x)
        x = self.classifier(x)
        return x
