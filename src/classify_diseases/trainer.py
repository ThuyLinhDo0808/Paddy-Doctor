import torch
import os
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # === Training Phase ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        # === Validation Phase ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / total
        epoch_val_acc = correct / total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"[{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {epoch_val_acc*100:.2f}%")
    # === Save history ===
    os.makedirs("checkpoints", exist_ok=True)
    history_path = os.path.join("checkpoints", "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"📊 Training history saved to {history_path}")
    # === Save Model ===
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/model_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f" Model saved to {save_path}")
    return history

def evaluate_model(model, dataloader, class_names, device):
    """
    Evaluates a PyTorch model on a given dataloader.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloader (DataLoader): Validation/test DataLoader
        class_names (list): List of class names, sorted
        device (torch.device): The device to run evaluation on
    """
    print(f"🖥️ Evaluating on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    model.eval()
    
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(" Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return {
        "accuracy": acc,
        "predictions": all_preds,
        "labels": all_labels,
        "class_names": class_names
    }
