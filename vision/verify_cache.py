
from pathlib import Path

import numpy as np

from .preprocess import sample_frames


ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIR = ROOT / "data" / "raw" / "all"
CACHE_DIR = ROOT / "data" / "processed" / "frames"


def main():
    cache_files = sorted(CACHE_DIR.glob("*.npy"))

    print("Cached files found:", len(cache_files))

    if not cache_files:
        raise RuntimeError("No cached files found")

    passed = 0
    failed = []

    for cache_path in cache_files:
        try:
            frames = np.load(
                cache_path,
                allow_pickle=False
            )

            assert frames.shape == (16, 224, 224, 3)
            assert frames.dtype == np.uint8

            assert frames.min() >= 0
            assert frames.max() <= 255

            video_path = VIDEO_DIR / (
                cache_path.stem + ".mp4"
            )

            assert video_path.exists(), (
                f"Original video missing: {video_path}"
            )

            # Compare cached frames against original
            original_frames = sample_frames(
                video_path,
                num_frames=16
            )

            assert np.array_equal(
                frames,
                original_frames
            ), "Cached frames do not match preprocessing"

            passed += 1

            print(
                f"PASSED: {cache_path.name} | "
                f"shape={frames.shape} | "
                f"dtype={frames.dtype}"
            )

        except Exception as error:
            failed.append((cache_path.name, str(error)))
            print(f"FAILED: {cache_path.name}: {error}")

    print("\nCache Verification Summary")
    print("--------------------------")
    print("Total:", len(cache_files))
    print("Passed:", passed)
    print("Failed:", len(failed))

    if failed:
        raise RuntimeError(
            f"{len(failed)} cached files failed verification"
        )

    print("\nAll cached arrays verified successfully!")


if __name__ == "__main__":
    main()
