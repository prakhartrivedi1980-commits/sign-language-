
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vision.dataset import LSA64Dataset
from vision.model import SignLanguageResNet


ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = ROOT / "checkpoints" / "vision" / "best.pt"

OUTPUT_DIR = (
    ROOT / "data" / "processed" / "resnet_features"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"ResNet checkpoint missing: {CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nResNet-18 Feature Extraction")
    print("----------------------------")
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)

    # Initialize model without downloading weights.
    model = SignLanguageResNet(
        num_classes=64,
        pretrained=False
    )

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)

    # Remove the 64-class classification layer.
    # The backbone now outputs 512 visual features.
    model.backbone.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_saved = 0
    total_skipped = 0

    with torch.inference_mode():
        for split in ("train", "val", "test"):
            dataset = LSA64Dataset(
                split=split,
                num_frames=16
            )

            split_dir = OUTPUT_DIR / split
            split_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            saved = 0
            skipped = 0

            for index in tqdm(
                range(len(dataset)),
                desc=f"Extracting {split}",
                unit="video"
            ):
                filename = dataset.videos[index].stem
                output_path = split_dir / f"{filename}.npy"

                if output_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                frames, _ = dataset[index]

                feature_chunks = []

                # Process small groups of frames on GPU.
                for start in range(
                    0, frames.shape[0], args.chunk_size
                ):
                    chunk = frames[
                        start:start + args.chunk_size
                    ].to(device)

                    features = model.backbone(chunk)

                    feature_chunks.append(
                        features.cpu().numpy()
                    )

                features = np.concatenate(
                    feature_chunks,
                    axis=0
                ).astype(np.float32)

                if features.shape != (16, 512):
                    raise ValueError(
                        f"Invalid feature shape for {filename}: "
                        f"{features.shape}"
                    )

                if not np.isfinite(features).all():
                    raise ValueError(
                        f"Non-finite features: {filename}"
                    )

                # Atomic save to avoid incomplete output files.
                temporary = output_path.with_suffix(".tmp")

                with open(temporary, "wb") as f:
                    np.save(f, features)

                os.replace(temporary, output_path)

                saved += 1

            total_saved += saved
            total_skipped += skipped

            print(
                f"{split}: saved={saved}, skipped={skipped}, "
                f"total={len(dataset)}"
            )

    print("\nFeature Extraction Summary")
    print("--------------------------")
    print("Saved:", total_saved)
    print("Skipped:", total_skipped)
    print("Total:", total_saved + total_skipped)
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
