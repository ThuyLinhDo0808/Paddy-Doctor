import torch
import torch.nn as nn
import torch.optim as optim
from src.age.trainer.base import BaseTrainer
import pandas as pd
import os
import time
from scripts.utils import EarlyStopping, get_device
from sklearn.metrics import mean_absolute_error, r2_score


def compute_regression_metrics(preds, targets):
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return mae, r2


class AgeTrainer(BaseTrainer):
    def __init__(self, model, config):
        device = get_device(force_cpu=config['train'].get('force_cpu', False))
        super().__init__(model, config, device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['train']['lr'])
        
        self.log.update({
            "train_loss": [], "train_mae": [], "train_r2": [],
            "val_mae": [], "val_r2": [], "val_loss": []
        })

    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        preds, targets = [], []
        sample_seen = 0
        batch_total = len(dataloader)
        log_interval = 50

        for batch_idx, (images, target) in enumerate(dataloader):
            images = images.to(self.device)
            target = target.float().to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images).squeeze()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            sample_seen += len(images)
            preds.extend(output.detach().cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == batch_total:
                avg_batch_loss = total_loss / (batch_idx + 1)
                print(f"  [Batch {batch_idx + 1}/{batch_total}] Avg Loss: {avg_batch_loss:.4f} | Samples: {sample_seen}")

        avg_loss = total_loss / batch_total
        mae, r2 = compute_regression_metrics(preds, targets)

        self.log['train_loss'].append(avg_loss)
        self.log['train_mae'].append(mae)
        self.log['train_r2'].append(r2)
        self.log['loss'].append(avg_loss)

        return avg_loss

    def _eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for images, target in dataloader:
                images = images.to(self.device)
                target = target.float().to(self.device)
                output = self.model(images).squeeze()
                loss = self.criterion(output, target)
                total_loss += loss.item()

                preds.extend(output.cpu().numpy().tolist())
                targets.extend(target.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataloader)
        mae, r2 = compute_regression_metrics(preds, targets)
        self.log['val_loss'].append(avg_loss)
        self.log['val_mae'].append(mae)
        self.log['val_r2'].append(r2)

        print(f"Validation Metrics -> MAE: {mae:.4f}, R²: {r2:.4f}")
        return avg_loss

    def fit(self, train_loader, val_loader):
        print("Training started...")
        epochs = self.config['train']['epochs']
        best_val = float('inf')
        start_time = time.time()
        early_stop = self.config['train'].get('early_stopping', False)
        stopper = EarlyStopping(patience=self.config['train'].get('patience', 5)) if early_stop else None

        model_name = self.config['train']['model_name']
        os.makedirs("checkpoints/age", exist_ok=True)
        best_model_path = f"checkpoints/age/{model_name}.pt"

        for epoch in range(epochs):
            print("----------------------------")
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_start = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._eval_epoch(val_loader)

            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start
            remaining = epoch_time * (epochs - epoch - 1)

            improvement = 0.0 if best_val == float('inf') else best_val - val_loss
            best_flag = "[Best]" if val_loss < best_val else ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), best_model_path)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} {best_flag}")
            print(f"Elapsed: {elapsed:.2f}s | Est. Remaining: {remaining:.2f}s | Improvement: {improvement:.4f}")

            if early_stop and stopper(val_loss):
                print("Early stopping triggered.")
                break

        torch.save(self.model.state_dict(), f"checkpoints/age/{model_name}.pt")

        csv_path = f"checkpoints/age/{model_name}.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)
        print(self.log)
        pd.DataFrame(self.log).to_csv(csv_path, index=False)
        print(f"Saved: checkpoints/age/{model_name}.pt, {csv_path}")
        if os.path.exists(best_model_path):
            print(f"Best model saved to {best_model_path}")
