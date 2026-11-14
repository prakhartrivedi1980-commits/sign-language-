
import time

import torch
import torch.nn as nn

from .model import SignLanguageResNet
from .chunked_training import chunked_video_backward


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    torch.manual_seed(42)

    model = SignLanguageResNet(
        num_classes=64,
        pretrained=False
    ).to(device)

    model.train()

    # Freeze BatchNorm statistics
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.amp.GradScaler("cuda")

    # One synthetic video with 16 frames
    frames = torch.randn(
        1, 16, 3, 224, 224,
        device=device
    )

    labels = torch.tensor([5], device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    loss, predictions = chunked_video_backward(
        model,
        frames,
        labels,
        chunk_size=4,
        loss_scale=1.0,
        use_amp=True,
        scaler=scaler
    )

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print("\nGPU Training Smoke Test")
    print("-----------------------")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loss:", round(loss, 4))
    print("Prediction:", predictions.item())
    print("Elapsed time:", round(elapsed, 2), "seconds")
    print("Peak allocated:", round(peak_allocated, 3), "GB")
    print("Peak reserved:", round(peak_reserved, 3), "GB")

    assert torch.isfinite(torch.tensor(loss))
    assert peak_reserved < 5.5

    print("\nGPU training smoke test PASSED!")


if __name__ == "__main__":
    main()
