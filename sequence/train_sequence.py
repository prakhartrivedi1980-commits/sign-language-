
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import SignLanguageLSTM


ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = ROOT / "data/processed/resnet_features"
CHECKPOINT_DIR = ROOT / "checkpoints/sequence"
HISTORY_PATH = ROOT / "results/sequence_history.csv"


class FeatureDataset(Dataset):
    def __init__(self, split):
        expected = {
            "train": 2240,
            "val": 320
        }

        if split not in expected:
            raise ValueError(f"Invalid split: {split}")

        self.files = sorted(
            (FEATURE_ROOT / split).glob("*.npy")
        )

        if len(self.files) != expected[split]:
            raise RuntimeError(
                f"{split}: expected {expected[split]} files, "
                f"found {len(self.files)}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        features = np.load(
            path,
            allow_pickle=False
        )

        if features.shape != (16, 512):
            raise ValueError(f"Invalid shape: {path.name}")

        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features: {path.name}")

        class_id = int(path.stem.split("_")[0])
        label = class_id - 1

        return (
            torch.from_numpy(
                features.astype(np.float32)
            ),
            torch.tensor(label, dtype=torch.long)
        )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False
    )

    for features, labels in progress:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        n = labels.size(0)

        total_loss += loss.item() * n
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total += n

        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{total_correct / total:.2%}"
        )

    return total_loss / total, total_correct / total


def main():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nLSA64 ResNet-18 + LSTM Training")
    print("--------------------------------")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = FeatureDataset("train")
    val_dataset = FeatureDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = SignLanguageLSTM(
        input_size=512,
        hidden_size=256,
        num_layers=2,
        num_classes=64,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    patience_counter = 0
    early_stopping_patience = 10
    max_epochs = 60

    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion,
            device, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.2%}",
            flush=True
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best.pt"
            )
        else:
            patience_counter += 1

        with open(
            HISTORY_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss",
                    "train_accuracy", "val_loss",
                    "val_accuracy"
                ]
            )
            writer.writeheader()
            writer.writerows(history)

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("\nLSTM training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print("Best model:", CHECKPOINT_DIR / "best.pt")


if __name__ == "__main__":
    main()
