
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import SignLanguageResNet


ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = ROOT / "data" / "processed" / "frames"

CHECKPOINT = ROOT / "checkpoints" / "vision" / "best.pt"

PREDICTIONS = (
    ROOT / "results" / "vision_evaluation"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "vision_evaluation"
    / "gradcam"
)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None

        # Final convolutional stage of ResNet-18
        self.target_layer = model.backbone.layer4[-1]

        self.handle = self.target_layer.register_forward_hook(
            self.save_activations
        )

    def save_activations(self, module, inputs, output):
        self.activations = output

    def generate(self, image_tensor, target_class):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # One frame: [1, 3, H, W]
        logits = self.model.backbone(image_tensor)

        score = logits[0, target_class]

        gradients = torch.autograd.grad(
            score,
            self.activations,
            retain_graph=False
        )[0]

        activations = self.activations.detach()

        # Average gradients over spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)

        return cam.detach().cpu().numpy()

    def close(self):
        self.handle.remove()


def normalize_frame(frame):
    image = frame.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std
    image = image.transpose(2, 0, 1)

    return torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0)


def create_overlay(frame, cam):
    heatmap = np.uint8(cam * 255)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        frame,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        default=None,
        help="Optional test video filename"
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=8
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)

    with open(PREDICTIONS, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Default: most confident incorrect prediction
    if args.filename is None:
        failures = [
            row for row in records
            if row["true_class"] != row["predicted_class"]
        ]

        failures.sort(
            key=lambda row: float(row["confidence"]),
            reverse=True
        )

        if not failures:
            raise RuntimeError("No failure cases found")

        selected = failures[0]

    else:
        matches = [
            row for row in records
            if row["filename"] == args.filename
        ]

        if not matches:
            raise ValueError("Filename not found in predictions")

        selected = matches[0]

    filename = selected["filename"]
    cache_path = CACHE_DIR / (
        Path(filename).stem + ".npy"
    )

    frames = np.load(cache_path, allow_pickle=False)

    if not 0 <= args.frame_index < len(frames):
        raise ValueError("Invalid frame index")

    frame = frames[args.frame_index]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SignLanguageResNet(
        num_classes=64,
        pretrained=False
    ).to(device)

    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    image_tensor = normalize_frame(frame).to(device)

    # CSV labels are 1-based; PyTorch labels are 0-based
    predicted_class = int(selected["predicted_class"]) - 1

    gradcam = GradCAM(model)

    try:
        cam = gradcam.generate(
            image_tensor,
            predicted_class
        )
    finally:
        gradcam.close()

    overlay = create_overlay(frame, cam)

    # Create comparison image
    combined = np.concatenate(
        [frame, overlay],
        axis=1
    )

    output_path = OUTPUT_DIR / (
        f"{Path(filename).stem}_frame"
        f"{args.frame_index:02d}_gradcam.png"
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    )

    print("\nGrad-CAM Visualization")
    print("----------------------")
    print("Video:", filename)
    print("True class:", selected["true_class"])
    print("Predicted class:", selected["predicted_class"])
    print("Frame index:", args.frame_index)
    print("Device:", device)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
