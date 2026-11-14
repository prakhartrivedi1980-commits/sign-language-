
import copy

import torch
import torch.nn.functional as F

from .model import SignLanguageResNet
from .chunked_training import chunked_video_backward


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")

    # Random weights avoid downloading ImageNet weights.
    model_standard = SignLanguageResNet(
        num_classes=64,
        pretrained=False
    ).to(device)

    model_chunked = copy.deepcopy(model_standard)

    # Evaluation mode fixes BatchNorm behavior.
    # Gradients still work without torch.no_grad().
    model_standard.eval()
    model_chunked.eval()

    frames = torch.randn(
        1, 4, 3, 64, 64,
        device=device
    )

    labels = torch.tensor([5], device=device)

    # Standard backpropagation
    logits = model_standard(frames)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    # Two-pass chunked backpropagation
    chunked_video_backward(
        model_chunked,
        frames,
        labels,
        chunk_size=2,
        loss_scale=1.0,
        use_amp=False
    )

    max_abs_error = 0.0
    max_rel_error = 0.0

    for (name_a, param_a), (name_b, param_b) in zip(
        model_standard.named_parameters(),
        model_chunked.named_parameters()
    ):
        assert name_a == name_b

        grad_a = param_a.grad
        grad_b = param_b.grad

        if grad_a is None and grad_b is None:
            continue

        assert grad_a is not None
        assert grad_b is not None

        abs_error = (grad_a - grad_b).abs().max().item()

        denominator = (
            grad_a.norm() + grad_b.norm()
        ).clamp_min(1e-12)

        rel_error = (
            (grad_a - grad_b).norm() / denominator
        ).item()

        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)

    print("Maximum absolute error:", max_abs_error)
    print("Maximum relative error:", max_rel_error)

    assert max_rel_error < 1e-4, (
        f"Gradient mismatch: {max_rel_error}"
    )

    print("Chunked gradient correctness PASSED!")


if __name__ == "__main__":
    main()
