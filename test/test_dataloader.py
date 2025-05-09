import sys
import os
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append(os.path.abspath("src"))

from dataloader import get_dataloaders

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataloader, val_dataloader = get_dataloaders(
    "data/train_images", "data/meta_train.csv", "label", transform=transform
)

# Plot a few images and labels from the dataloader
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (images, labels) in enumerate(train_dataloader):
    if i == 2:  # Display only the first 10 images
        break
    for j in range(5):
        axes[i, j].imshow(
            images[j].permute(1, 2, 0)
        )  # Convert to HWC format for display
        axes[i, j].set_title(f"Label: {labels[j]}")
        axes[i, j].axis("off")
plt.tight_layout()
plt.show()
