
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import SignLanguageLSTM


ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "data/processed/resnet_features/test"
CHECKPOINT = ROOT / "checkpoints/sequence/best.pt"
OUTPUT_DIR = ROOT / "results/sequence_evaluation"


class TestDataset(Dataset):
    def __init__(self):
        self.files = sorted(FEATURE_DIR.glob("*.npy"))

        if len(self.files) != 640:
            raise RuntimeError(
                f"Expected 640 files, found {len(self.files)}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        features = np.load(path, allow_pickle=False)

        if features.shape != (16, 512):
            raise ValueError(f"Invalid features: {path.name}")

        label = int(path.stem.split("_")[0]) - 1

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
            path.stem
        )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SignLanguageLSTM().to(device)

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = TestDataset()

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    true_labels = []
    predicted_labels = []
    records = []
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss(
        reduction="sum"
    )

    with torch.inference_mode():
        for features, labels, names in tqdm(
            loader,
            desc="Evaluating LSTM"
        ):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += criterion(logits, labels).item()

            for i, name in enumerate(names):
                actual = int(labels[i].item())
                predicted = int(predictions[i].item())

                true_labels.append(actual)
                predicted_labels.append(predicted)

                records.append({
                    "filename": name + ".mp4",
                    "true_class": actual + 1,
                    "predicted_class": predicted + 1,
                    "confidence": float(
                        probabilities[i, predicted].item()
                    ),
                    "correct": actual == predicted
                })

    accuracy = accuracy_score(
        true_labels, predicted_labels
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(range(64)),
            average="macro",
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(64))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "test_videos": len(dataset),
        "accuracy": float(accuracy),
        "loss": total_loss / len(dataset),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "correct_predictions": int(
            sum(a == b for a, b in zip(
                true_labels, predicted_labels
            ))
        )
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    with open(
        OUTPUT_DIR / "test_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "true_class",
                "predicted_class", "confidence",
                "correct"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    np.save(
        OUTPUT_DIR / "confusion_matrix.npy",
        matrix
    )

    print("\nLSTM Test Results")
    print("-----------------")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(
        "Correct predictions:",
        results["correct_predictions"]
    )
    print(
        "Incorrect predictions:",
        len(dataset) - results["correct_predictions"]
    )


if __name__ == "__main__":
    main()
