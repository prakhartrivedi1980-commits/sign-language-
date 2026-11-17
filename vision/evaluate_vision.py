
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LSA64Dataset
from .model import SignLanguageResNet


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/vision.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vision/best.pt"
    )

    args = parser.parse_args()

    with open(
        ROOT / args.config,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nLSA64 ResNet-18 Test Evaluation")
    print("--------------------------------")
    print("Device:", device)

    checkpoint_path = ROOT / args.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    test_dataset = LSA64Dataset(
        split="test",
        num_frames=config["dataset"]["num_frames"]
    )

    if len(test_dataset) != 640:
        raise RuntimeError(
            f"Expected 640 test videos, got {len(test_dataset)}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = SignLanguageResNet(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False
    ).to(device)

    # best.pt contains only the model state_dict.
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Test videos:", len(test_dataset))

    chunk_size = config["training"]["frame_chunk_size"]

    true_labels = []
    predicted_labels = []
    probability_rows = []
    prediction_records = []

    total_loss = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        for index, (frames, labels) in enumerate(
            tqdm(
                test_loader,
                desc="Evaluating test videos",
                unit="video"
            )
        ):
            frames = frames.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Memory-efficient inference.
            logits = model.predict_chunked(
                frames,
                chunk_size=chunk_size
            )

            loss = F.cross_entropy(
                logits,
                labels
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            true_label = int(labels.item())
            predicted_label = int(predictions.item())

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            probability_rows.append(
                probabilities[0].cpu().numpy()
            )

            prediction_records.append({
                "filename": test_dataset.videos[index].name,
                "true_class": true_label + 1,
                "predicted_class": predicted_label + 1,
                "confidence": float(confidence.item()),
                "correct": true_label == predicted_label
            })

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predicted_labels)
    probabilities = np.stack(probability_rows)

    num_classes = config["dataset"]["num_classes"]
    class_ids = list(range(num_classes))

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=[
            f"Class_{i:03d}"
            for i in range(1, num_classes + 1)
        ],
        zero_division=0,
        output_dict=True
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "test_videos": len(test_dataset),
        "test_signers": [9, 10],
        "accuracy": float(accuracy),
        "loss": float(total_loss / total_samples),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "correct_predictions": int(
            np.sum(y_true == y_pred)
        ),
        "incorrect_predictions": int(
            np.sum(y_true != y_pred)
        ),
        "evaluation_seconds": float(elapsed)
    }

    output_dir = ROOT / "results" / "vision_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics.
    with open(
        output_dir / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    # Save classification report.
    with open(
        output_dir / "classification_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    # Save individual video predictions.
    with open(
        output_dir / "test_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "true_class",
                "predicted_class",
                "confidence",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(prediction_records)

    # Save confusion matrix as NumPy array.
    np.save(
        output_dir / "confusion_matrix.npy",
        matrix
    )

    # Save confusion matrix figure.
    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(image, ax=ax)

    ax.set_title(
        "LSA64 ResNet-18 Test Confusion Matrix"
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))

    ax.set_xticklabels(
        range(1, num_classes + 1),
        fontsize=5,
        rotation=90
    )

    ax.set_yticklabels(
        range(1, num_classes + 1),
        fontsize=5
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=180
    )

    plt.close(fig)

    print("\nTest Evaluation Results")
    print("-----------------------")
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
        results["incorrect_predictions"]
    )
    print(f"Evaluation time: {elapsed:.1f}s")
    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
