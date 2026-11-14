
import torch
import torch.nn.functional as F


def chunked_video_backward(
    model,
    frames,
    labels,
    chunk_size=4,
    loss_scale=1.0,
    use_amp=True,
    scaler=None,
):
    """
    Compute video-level cross-entropy gradients
    while processing small frame chunks.

    Assumes model.backbone is a ResNet classifier.

    For training, BatchNorm running statistics must
    remain frozen so both passes use consistent
    normalization behavior.
    """
    if frames.ndim != 5:
        raise ValueError("Expected [B, T, C, H, W]")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch_size, num_frames, channels, height, width = frames.shape

    if num_frames <= 0:
        raise ValueError("Video must contain frames")

    device_type = frames.device.type
    amp_enabled = use_amp and device_type == "cuda"

    def autocast_context():
        return torch.autocast(
            device_type=device_type,
            enabled=amp_enabled,
        )

    # PASS 1: Compute video logits without graphs.
    with torch.no_grad():
        video_logits = None

        for start in range(0, num_frames, chunk_size):
            chunk = frames[:, start:start + chunk_size]
            chunk_count = chunk.shape[1]

            chunk = chunk.reshape(
                -1, channels, height, width
            )

            with autocast_context():
                logits = model.backbone(chunk)

            chunk_sum = logits.reshape(
                batch_size, chunk_count, -1
            ).float().sum(dim=1)

            video_logits = (
                chunk_sum if video_logits is None
                else video_logits + chunk_sum
            )

        video_logits = video_logits / num_frames

    # Gradient of the video loss with respect
    # to the aggregated video logits.
    with torch.enable_grad():
        detached_logits = (
            video_logits.detach().requires_grad_(True)
        )

        loss = F.cross_entropy(
            detached_logits,
            labels
        )

        video_gradient = torch.autograd.grad(
            loss,
            detached_logits
        )[0].detach()

    # PASS 2: Recompute each chunk and backpropagate
    # its contribution to the video-level gradient.
    for start in range(0, num_frames, chunk_size):
        chunk = frames[:, start:start + chunk_size]
        chunk_count = chunk.shape[1]

        chunk = chunk.reshape(
            -1, channels, height, width
        )

        with autocast_context():
            logits = model.backbone(chunk)

        chunk_logits = logits.reshape(
            batch_size, chunk_count, -1
        ).float().sum(dim=1) / num_frames

        surrogate = (
            chunk_logits * video_gradient
        ).sum() * loss_scale

        if scaler is not None and amp_enabled:
            scaler.scale(surrogate).backward()
        else:
            surrogate.backward()

    return (
        loss.detach().item(),
        video_logits.argmax(dim=1).detach()
    )
