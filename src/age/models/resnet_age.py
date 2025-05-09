import torch.nn as nn
from torchvision.models import resnet18

class ResNetAge(nn.Module):
    def __init__(self):
        super(ResNetAge, self).__init__()
        base_model = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # remove original fc
        # eeps all convolutional layers + global average pooling
        # Outputs a 512-dimensional embedding per image
        # Learn to map image features to predict age
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output single value for age
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x
