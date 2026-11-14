
from pathlib import Path

import cv2
import numpy as np



def sample_frames(video_path, num_frames=16):
    """
    Sample evenly spaced frames using sequential decoding.

    Returns:
        Array of shape (num_frames, 224, 224, 3).
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise ValueError("Video has no frames")

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
            dtype=int
        )

        required = set(indices.tolist())
        selected = {}

        for frame_index in range(max(required) + 1):
            success, frame = cap.read()

            if not success or frame is None:
                raise ValueError(
                    f"Cannot decode frame {frame_index}"
                )

            if frame_index in required:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame = cv2.resize(
                    frame,
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR
                )

                selected[frame_index] = frame

        frames = [
            selected[int(index)]
            for index in indices
        ]

        return np.stack(frames)

    finally:
        cap.release()


def normalize_frames(frames):
    """
    Apply ImageNet normalization.

    Input shape: (T, H, W, 3)
    Output shape: (T, 3, H, W)
    """
    frames = frames.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    frames = (frames - mean) / std

    # Convert channel-last to channel-first
    frames = np.transpose(
        frames,
        (0, 3, 1, 2)
    )

    return frames.astype(np.float32)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    video_path = (
        project_root
        / "data"
        / "raw"
        / "all"
        / "001_001_001.mp4"
    )

    frames = sample_frames(video_path)
    normalized = normalize_frames(frames)

    print("Original sampled frames:", frames.shape)
    print("Normalized frames:", normalized.shape)
    print("Data type:", normalized.dtype)
    print("Minimum value:", normalized.min())
    print("Maximum value:", normalized.max())
