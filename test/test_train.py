import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score


sys.path.append(os.path.abspath("src"))

from dataloader import get_dataloaders
from variety_id.trainer import Trainer
from variety_id.models.cbam import CBAMResNet18

if __name__ == "__main__":
    device = torch.device("cuda")

    # ----- Define transforms -----
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # ----- Load data -----
    train_dataloader, val_dataloader = get_dataloaders(
        "data/train_images",
        "data/meta_train.csv",
        "label",
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=32,
        oversample=True,
    )

    # ----- Initialize model -----
    model = CBAMResNet18(num_classes=10)
    model = model.to(device)

    # ----- Define loss, optimizer, scheduler, and metric -----
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    metric = MulticlassF1Score(num_classes=10, average="weighted").to(device)

    # ----- Run training -----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
        device=device,
        model_name="MixupCBAMResNet18_cnn_2",
        save=True,
        mixup=True,
    )

    trainer.fit(train_dataloader, val_dataloader, epochs=100)
