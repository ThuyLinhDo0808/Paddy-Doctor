import matplotlib.pyplot as plt
import os
import json
def print_label_distribution(label_counter, class_names):
    print("\n Label Distribution:")
    for idx, count in label_counter.items():
        print(f"  [{idx}] {class_names[idx]}: {count} images")

def plot_label_distribution(label_counter, class_names):
    plt.figure(figsize=(10, 5))
    plt.bar([class_names[i] for i in label_counter.keys()],
            [label_counter[i] for i in label_counter.keys()],
            color="skyblue")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_size_distribution(size_counter):
    plt.figure(figsize=(10, 5))
    plt.bar([f"{w}x{h}" for (h, w) in size_counter.keys()],
            list(size_counter.values()),
            color="salmon")
    plt.title("Image Size Distribution (After Transform)")
    plt.xlabel("Size (HxW)")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.show()

def plot_all_histories_in_dir(history_dir: str):
    """
    Loads and plots all training history files (.json) in the specified directory.
    Each file should contain 'train_loss', 'val_loss', 'train_acc', and 'val_acc'.

    Args:
        history_dir (str): Directory containing JSON history files.
    """
    history_files = [f for f in os.listdir(history_dir) if f.endswith(".json")]
    
    if not history_files:
        print(f"❌ No JSON history files found in {history_dir}")
        return

    print(f"📂 Found {len(history_files)} history files in '{history_dir}': {history_files}")

    plt.figure(figsize=(14, 6))

    # === Subplot 1: Loss Curves ===
    plt.subplot(1, 2, 1)
    for filename in history_files:
        path = os.path.join(history_dir, filename)
        with open(path, "r") as f:
            history = json.load(f)
        epochs = range(1, len(history["train_loss"]) + 1)
        label = os.path.splitext(filename)[0]
        plt.plot(epochs, history["train_loss"], label=f"{label} - Train")
        plt.plot(epochs, history["val_loss"], linestyle="--", label=f"{label} - Val")

    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # === Subplot 2: Accuracy Curves ===
    plt.subplot(1, 2, 2)
    for filename in history_files:
        path = os.path.join(history_dir, filename)
        with open(path, "r") as f:
            history = json.load(f)
        epochs = range(1, len(history["train_acc"]) + 1)
        label = os.path.splitext(filename)[0]
        plt.plot(epochs, [x * 100 for x in history["train_acc"]], label=f"{label} - Train")
        plt.plot(epochs, [x * 100 for x in history["val_acc"]], linestyle="--", label=f"{label} - Val")

    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_all_histories_in_dir("checkpoints")