
from pathlib import Path
import csv
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "all"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "video_verification.csv"

def verify_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    result = {
        "filename": video_path.name,
        "opened": False,
        "readable": False,
        "frames": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "status": "FAILED"
    }

    try:
        if not cap.isOpened():
            return result

        result["opened"] = True

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        result["frames"] = frames
        result["width"] = width
        result["height"] = height
        result["fps"] = round(fps, 2)

        if frames <= 0 or width <= 0 or height <= 0:
            return result

        if fps <= 0:
            return result

        # Check first, middle, and last frames
        positions = sorted(set([
            0,
            frames // 2,
            frames - 1
        ]))

        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            success, frame = cap.read()

            if not success or frame is None:
                return result

            if frame.size == 0:
                return result

        result["readable"] = True
        result["status"] = "PASSED"
        return result

    finally:
        cap.release()


def main():
    videos = sorted(DATASET_DIR.glob("*.mp4"))

    print(f"Found {len(videos)} videos")

    if len(videos) != 3200:
        raise ValueError(
            f"Expected 3200 videos, found {len(videos)}"
        )

    results = []

    for index, video in enumerate(videos, start=1):
        result = verify_video(video)
        results.append(result)

        if index % 100 == 0:
            print(f"Verified {index}/{len(videos)} videos")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename", "opened", "readable",
                "frames", "width", "height",
                "fps", "status"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    passed = sum(r["status"] == "PASSED" for r in results)
    failed = len(results) - passed

    print("\nVideo Verification Summary")
    print("--------------------------")
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        print("\nFailed files:")
        for r in results:
            if r["status"] == "FAILED":
                print(r["filename"])

    print(f"\nReport saved to: {OUTPUT_CSV}")

    if failed:
        raise RuntimeError(
            f"{failed} videos failed verification"
        )


if __name__ == "__main__":
    main()
