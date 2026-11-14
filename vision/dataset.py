
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from .preprocess import normalize_frames


ROOT = Path(__file__).resolve().parent.parent


class LSA64Dataset(Dataset):
    def __init__(self, split="train", num_frames=16):
        config_path = ROOT / "configs" / "split.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        if num_frames != 16:
            raise ValueError(
                "Current frame cache contains exactly 16 frames"
            )

        self.num_frames = num_frames

        cache_root = ROOT / "data" / "processed" / "frames"

        if not cache_root.exists():
            raise FileNotFoundError(cache_root)

        split_key = {
            "train": "train_signers",
            "val": "val_signers",
            "test": "test_signers"
        }[split]

        allowed_signers = set(
            config["split"][split_key]
        )

        self.videos = []
        self.labels = []

        for path in sorted(cache_root.glob("*.npy")):
            parts = path.stem.split("_")

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid cache filename: {path.name}"
                )

            class_id, signer_id, repetition = map(
                int, parts
            )

            if not (
                1 <= class_id <= 64
                and 1 <= signer_id <= 10
                and 1 <= repetition <= 5
            ):
                raise ValueError(
                    f"Invalid identifiers: {path.name}"
                )

            if signer_id in allowed_signers:
                self.videos.append(path)
                self.labels.append(class_id - 1)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        cache_path = self.videos[index]

        frames = np.load(
            cache_path,
            allow_pickle=False
        )

        if frames.shape != (16, 224, 224, 3):
            raise ValueError(
                f"Invalid shape: {cache_path.name}"
            )

        frames = normalize_frames(frames)

        frames = torch.from_numpy(
            np.ascontiguousarray(frames)
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return frames, label


if __name__ == "__main__":
    train = LSA64Dataset("train")
    val = LSA64Dataset("val")
    test = LSA64Dataset("test")

    print("Training videos:", len(train))
    print("Validation videos:", len(val))
    print("Testing videos:", len(test))

    assert len(train) == 2240
    assert len(val) == 320
    assert len(test) == 640

    frames, label = train[0]

    print("Single sample:", frames.shape)
    print("Label:", label.item())
    print("Data type:", frames.dtype)

    loader = DataLoader(
        train,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    batch_frames, batch_labels = next(iter(loader))

    print("Batch frames:", batch_frames.shape)
    print("Batch labels:", batch_labels.shape)

    assert batch_frames.shape == (
        2, 16, 3, 224, 224
    )

    print("Cached dataset loader PASSED!")
