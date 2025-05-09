import pandas as pd
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset
from PIL import Image
import os

def get_balanced_dataframe(df: pd.DataFrame, label_col: str, strategy="weights"):
    label_counts = df[label_col].value_counts()

    if strategy == "weights":
        weights = compute_class_weight(class_weight="balanced",
                                       classes=label_counts.index,
                                       y=df[label_col])
        return tensor(weights, dtype=torch.float), None  # weights_tensor, df not changed

    elif strategy == "oversample":
        max_count = label_counts.max()
        balanced_df = pd.concat([
            resample(df[df[label_col] == cls], replace=True, n_samples=max_count, random_state=42)
            for cls in label_counts.index
        ])
        return None, balanced_df.reset_index(drop=True)

    elif strategy == "undersample":
        min_count = label_counts.min()
        balanced_df = pd.concat([
            resample(df[df[label_col] == cls], replace=False, n_samples=min_count, random_state=42)
            for cls in label_counts.index
        ])
        return None, balanced_df.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

class ImageCSVLoader(Dataset):
    def __init__(self, df, image_dir, transform=None, label_col="label"):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.label_col = label_col
        self.class_names = sorted(self.df[label_col].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 👇 Construct: /data/train_images/<label>/<image_id>
        label_str = str(row[self.label_col])
        img_path = os.path.join(self.image_dir, label_str, row["image_id"])
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[label_str]
        return image, label

