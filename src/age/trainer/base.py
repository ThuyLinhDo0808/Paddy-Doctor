import torch
import os
import pandas as pd
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for all task-specific trainers.
    Handles core trainer interface, logging, and checkpointing.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)

        self.metrics = {}  # define in subclass 
        self.log = {
            "loss": [],
            "val_loss": [],
            "lr": [],
        }

    @abstractmethod
    def _train_epoch(self, dataloader):
        pass

    @abstractmethod
    def _eval_epoch(self, dataloader):
        pass

    @abstractmethod
    def fit(self, train_loader, val_loader):
        pass

    def load_checkpoint(self):
        name = self.config['model']['name']
        ckpt_path = f"checkpoints/{name}.pt"
        log_path = f"logs/{name}.csv"

        if os.path.exists(log_path):
            self.log = pd.read_csv(log_path).to_dict(orient="list")

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.model.to(self.device)
            print(f" Loaded model from {ckpt_path}")
            return True

        return False