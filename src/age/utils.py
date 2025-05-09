import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics(log_path, output_dir="checkpoints/age"):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)

    metrics = [
        ("train_loss", "val_loss"),
        ("train_mae","val_mae"),
        ("train_rmse","val_rmse",),
        ("train_r2","val_r2",)
    ]

    for metric_group in metrics:
        plt.figure()
        for metric in metric_group:
            if metric in df:
                plt.plot(df[metric], label=metric)

        plt.title(" / ".join(metric_group))
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        name = "_".join(metric_group) + ".png"
        plt.savefig(os.path.join(output_dir, name))
        plt.close()

    print(f"Metric plots saved to {output_dir}")