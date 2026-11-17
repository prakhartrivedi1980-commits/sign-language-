
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "results" / "vision_history.csv"
OUTPUT_DIR = ROOT / "results" / "vision_evaluation"


def main():
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(HISTORY_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(HISTORY_FILE)

    required = [
        "epoch",
        "stage",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ]

    for column in required:
        if column not in history.columns:
            raise ValueError(
                f"Missing history column: {column}"
            )

    epochs = history["epoch"]

    # Training and validation loss
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    ax.plot(
        epochs,
        history["val_loss"],
        marker="s",
        label="Validation Loss"
    )

    ax.set_title("ResNet-18 Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=180
    )
    plt.close(fig)

    # Training and validation accuracy
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        history["train_accuracy"] * 100,
        marker="o",
        label="Training Accuracy"
    )

    ax.plot(
        epochs,
        history["val_accuracy"] * 100,
        marker="s",
        label="Validation Accuracy"
    )

    ax.set_title("ResNet-18 Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "training_accuracy.png",
        dpi=180
    )
    plt.close(fig)

    # Report best validation epoch
    best_index = history["val_accuracy"].idxmax()
    best_row = history.loc[best_index]

    print("\nResNet-18 Training Analysis")
    print("---------------------------")
    print("Total epochs:", len(history))
    print("Best epoch:", int(best_row["epoch"]))
    print(
        "Best validation accuracy:",
        f"{best_row['val_accuracy']:.2%}"
    )
    print(
        "Final training accuracy:",
        f"{history['train_accuracy'].iloc[-1]:.2%}"
    )

    print("\nPlots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
