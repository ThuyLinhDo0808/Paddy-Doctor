import json
import matplotlib.pyplot as plt
import os


def plot_training_history(history_path: str):
    """
    Plots training and validation loss/accuracy curves from a saved training history file.

    Args:
        history_path (str): Path to the JSON file containing training history.
    """
    if not os.path.exists(history_path):
        print(f"❌ File not found: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # === Plot Loss ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # === Plot Accuracy ===
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs, [x * 100 for x in history["train_acc"]], label="Train Acc", marker="o"
    )
    plt.plot(epochs, [x * 100 for x in history["val_acc"]], label="Val Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
