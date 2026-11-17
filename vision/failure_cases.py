
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS = (
    ROOT / "results" / "vision_evaluation"
    / "test_predictions.csv"
)

CACHE_DIR = (
    ROOT / "data" / "processed" / "frames"
)

OUTPUT_DIR = (
    ROOT / "results" / "vision_evaluation"
)

def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS)

    # Keep only incorrect predictions.
    failures = df[
        df["true_class"] != df["predicted_class"]
    ].copy()

    # Highest-confidence incorrect predictions first.
    failures = failures.sort_values(
        "confidence",
        ascending=False
    )

    print("Total incorrect predictions:", len(failures))

    selected = failures.head(10)

    fig, axes = plt.subplots(
        5, 2,
        figsize=(12, 18)
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(
        axes, selected.iterrows()
    ):
        filename = Path(row["filename"]).stem + ".npy"
        cache_path = CACHE_DIR / filename

        frames = np.load(
            cache_path,
            allow_pickle=False
        )

        # Display middle frame of the video.
        middle_frame = frames[len(frames) // 2]

        ax.imshow(middle_frame)

        ax.set_title(
            f"True: {int(row['true_class']):03d} | "
            f"Predicted: {int(row['predicted_class']):03d}\n"
            f"Confidence: {row['confidence']:.2%}",
            fontsize=10
        )

        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(
        "LSA64 ResNet-18: High-Confidence Failure Cases",
        fontsize=15
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "failure_cases.png"

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Failure-case gallery saved to:", output_path)


if __name__ == "__main__":
    main()
