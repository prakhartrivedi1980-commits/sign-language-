
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class SignLanguageResNet(nn.Module):
    def __init__(self, num_classes=64, pretrained=True):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Standard differentiable forward pass.

        Input:  [B, T, 3, H, W]
        Output: [B, 64]
        """
        if x.ndim != 5:
            raise ValueError(
                "Expected [B, T, C, H, W]"
            )

        batch_size, num_frames, channels, height, width = x.shape

        if channels != 3:
            raise ValueError("Expected 3 RGB channels")

        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        frame_logits = self.backbone(x)

        video_logits = frame_logits.reshape(
            batch_size,
            num_frames,
            self.num_classes
        ).mean(dim=1)

        return video_logits

    @torch.no_grad()
    def predict_chunked(self, x, chunk_size=4):
        """
        Memory-efficient inference.

        This method is for validation/inference only,
        not for gradient-based training.
        """
        if x.ndim != 5:
            raise ValueError("Expected [B, T, C, H, W]")

        batch_size, num_frames, channels, height, width = x.shape

        if channels != 3:
            raise ValueError("Expected 3 RGB channels")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        frame_logits = []

        for start in range(0, num_frames, chunk_size):
            chunk = x[:, start:start + chunk_size]

            chunk = chunk.reshape(
                -1, channels, height, width
            )

            logits = self.backbone(chunk)

            logits = logits.reshape(
                batch_size, -1, self.num_classes
            )

            frame_logits.append(logits)

        return torch.cat(
            frame_logits, dim=1
        ).mean(dim=1)


if __name__ == "__main__":
    model = SignLanguageResNet(
        num_classes=64,
        pretrained=False
    )

    model.eval()

    dummy_input = torch.randn(
        1, 2, 3, 224, 224
    )

    with torch.inference_mode():
        output = model(dummy_input)

        chunked_output = model.predict_chunked(
            dummy_input,
            chunk_size=1
        )

    print("Input shape:", dummy_input.shape)
    print("Standard output:", output.shape)
    print("Chunked output:", chunked_output.shape)

    assert output.shape == (1, 64)
    assert chunked_output.shape == (1, 64)

    assert torch.allclose(
        output,
        chunked_output,
        atol=1e-4,
        rtol=1e-4
    )

    print("ResNet-18 model tests PASSED!")
