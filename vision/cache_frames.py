
from pathlib import Path
import argparse
import os

import numpy as np
from tqdm import tqdm

from .preprocess import sample_frames


ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw" / "all"
CACHE_DIR = ROOT / "data" / "processed" / "frames"


def cache_video(video_path, overwrite=False):
    output_path = CACHE_DIR / f"{video_path.stem}.npy"

    if output_path.exists() and not overwrite:
        return "skipped"

    frames = sample_frames(
        video_path,
        num_frames=16
    )

    if frames.shape != (16, 224, 224, 3):
        raise ValueError(
            f"Unexpected shape for {video_path.name}: "
            f"{frames.shape}"
        )

    if frames.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 frames: {video_path.name}"
        )

    # Save atomically so interrupted writes don't
    # leave a valid-looking incomplete cache file.
    temporary_path = output_path.with_suffix(".tmp")

    with open(temporary_path, "wb") as f:
        np.save(f, frames)

    os.replace(temporary_path, output_path)

    return "saved"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if len(videos) != 3200:
        raise RuntimeError(
            f"Expected 3200 videos, found {len(videos)}"
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        videos = videos[:args.limit]

    saved = 0
    skipped = 0
    failed = []

    for video in tqdm(
        videos,
        desc="Caching LSA64 frames",
        unit="video"
    ):
        try:
            result = cache_video(
                video,
                overwrite=args.overwrite
            )

            if result == "saved":
                saved += 1
            else:
                skipped += 1

        except Exception as error:
            failed.append((video.name, str(error)))
            tqdm.write(
                f"FAILED: {video.name}: {error}"
            )

    print("\nFrame Cache Summary")
    print("-------------------")
    print("Saved:", saved)
    print("Skipped:", skipped)
    print("Failed:", len(failed))
    print("Cache location:", CACHE_DIR)

    if failed:
        raise RuntimeError(
            f"{len(failed)} videos failed caching"
        )


if __name__ == "__main__":
    main()
