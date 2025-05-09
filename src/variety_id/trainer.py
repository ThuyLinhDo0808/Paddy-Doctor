import time
import torch
import pandas as pd
from timm.data import Mixup


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None  # Store the best model

    def early_stop(self, monitor_metric, model):
        if monitor_metric < self.min_validation_loss:
            self.min_validation_loss = monitor_metric
            self.counter = 0
            self.best_model_state = model.state_dict()  # Save the best model state
        elif monitor_metric > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


mixup_fn = Mixup(
    mixup_alpha=0.4,  # Mixup strength
    cutmix_alpha=1.0,  # CutMix strength
    cutmix_minmax=None,
    prob=1.0,  # Probability to apply (1.0 = always)
    switch_prob=0.5,  # Mixup or CutMix
    mode="batch",  # 'batch' works best
    label_smoothing=0.1,  # same as in your loss
    num_classes=10,
)


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        metric,
        device,
        model_name,
        scheduler=None,
        save=True,
        mixup=False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.scheduler = scheduler
        self.save = save
        self.model_name = model_name
        self.mixup = mixup

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "lr": [],
        }

    def train_epoch(self, dataloader):
        start = time.time()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_loss = 0

        self.metric.reset()
        self.model.train()

        for batch_idx, (img, variety) in enumerate(dataloader):
            img, variety = img.to(self.device), variety.to(self.device)

            if self.mixup:
                img, variety = mixup_fn(img, variety)

            pred_variety = self.model(img)

            loss = self.loss_fn(pred_variety, variety)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if variety.ndim == 2:  # Mixup target (soft labels)
                target_labels = variety.argmax(dim=1)
            else:
                target_labels = variety

            self.metric(pred_variety.argmax(1), target_labels)

            if batch_idx % 100 == 0:
                loss_val = loss.item()
                current = batch_idx * len(img)
                print(f"Loss: {loss_val:>7f} [{current:>5d}/{size:>5d}]")

        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        f1_score = self.metric.compute().item()
        lr = self.optimizer.param_groups[0]["lr"]

        end = time.time()

        self.history["train_loss"].append(avg_loss)
        self.history["train_f1"].append(f1_score)
        self.history["lr"].append(lr)

        print(
            f"Train summary: [{lr}] {avg_loss:.4f} | {f1_score:.4f} | {end - start:.2f}s"
        )

    def val_epoch(self, dataloader):
        start = time.time()
        num_batches = len(dataloader)
        total_loss = 0

        self.metric.reset()
        self.model.eval()

        with torch.no_grad():
            for img, variety in dataloader:
                img, variety = img.to(self.device), variety.to(self.device)

                pred_variety = self.model(img)
                loss = self.loss_fn(pred_variety, variety)
                total_loss += loss.item()

                self.metric(pred_variety.argmax(1), variety)

        avg_loss = total_loss / num_batches
        f1_score = self.metric.compute().item()

        end = time.time()

        self.history["val_loss"].append(avg_loss)
        self.history["val_f1"].append(f1_score)

        print(
            f"Validation summary: {avg_loss:.4f} | {f1_score:.4f} | {end - start:.2f}s"
        )

    def fit(self, train_dataloader, val_dataloader, epochs):
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_epoch(train_dataloader)
            self.val_epoch(val_dataloader)

            if early_stopper.early_stop(self.history["val_loss"][-1], self.model):
                print("Early stopping")
                break

        if self.save:
            torch.save(
                early_stopper.best_model_state,
                self.model_name + ".pt",
            )
            pd.DataFrame(self.history).to_csv(self.model_name + ".csv", index=False)

        print("Training complete!")

        return self.history
