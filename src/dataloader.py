import os
import pandas as pd
from PIL import Image
import torch
from torchvision.io import decode_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class RiceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        labels_path: str,
        label_type: Literal["label", "variety", "age"],
        split: Literal["train", "val"] = "train",
        transform=None,
        target_transform=None,
        val_size: float = 0.2,
        random_seed: int = 42,
        oversample: bool = False,
    ):
        self.image_dir = image_dir
        self.label_type = label_type
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(labels_path)

        # Split the full dataset into train and val sets
        train_df, val_df = train_test_split(
            df, test_size=val_size, random_state=random_seed, stratify=df[label_type]
        )
        self.metadata = train_df if     split == "train" else val_df

        if oversample and split == "train":
            # Perform oversampling
            class_dfs = []
            max_size = self.metadata[label_type].value_counts().max()
            for class_label, group in self.metadata.groupby(label_type):
                upsampled = resample(
                    group,
                    replace=True,
                    n_samples=max_size,
                    random_state=random_seed,
                )
                class_dfs.append(upsampled)
            self.metadata = (
                pd.concat(class_dfs)
                .sample(frac=1, random_state=random_seed)
                .reset_index(drop=True)
            )

        # Build image paths and targets
        self.image_paths = []
        self.targets = []

        self.classes = sorted(self.metadata[self.label_type].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for _, row in self.metadata.iterrows():
            label_folder = row["label"]  # folder is based on label column
            image_id = row["image_id"]
            image_path = os.path.join(image_dir, label_folder, image_id)

            self.image_paths.append(image_path)
            self.targets.append(row[label_type])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = decode_image(self.image_paths[idx])
        image = to_pil_image(image)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        # Convert class string to index
        target = self.class_to_idx[target]

        if self.target_transform:
            target = self.target_transform(target)
        else:
            target = torch.tensor(target, dtype=torch.long)

        return image, target


def get_dataloaders(
    image_dir: str,
    labels_path: str,
    label_type: Literal["label", "variety", "age"],
    batch_size: int = 32,
    val_size: float = 0.2,
    random_seed: int = 42,
    train_transform=None,
    val_transform=None,
    target_transform=None,
    oversample: bool = False,
):
    train_ds = RiceDataset(
        image_dir=image_dir,
        labels_path=labels_path,
        label_type=label_type,
        split="train",
        transform=train_transform,
        target_transform=target_transform,
        val_size=val_size,
        random_seed=random_seed,
        oversample=oversample,
    )
    val_ds = RiceDataset(
        image_dir=image_dir,
        labels_path=labels_path,
        label_type=label_type,
        split="val",
        transform=val_transform,
        target_transform=target_transform,
        val_size=val_size,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader
