
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import LSA64Dataset
from .model import SignLanguageResNet
from .chunked_training import chunked_video_backward


ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_stage(model, stage):
    # Freeze backbone during head training.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = stage != "head"

    # Always train classification head.
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad = True


# --------------------------------------------------
# Validation
# --------------------------------------------------

def evaluate(model, loader, device, chunk_size):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc="Validation",
            unit="batch",
            leave=False
        )

        for frames, labels in progress:
            frames = frames.to(
                device, non_blocking=True
            )
            labels = labels.to(
                device, non_blocking=True
            )

            logits = model.predict_chunked(
                frames,
                chunk_size=chunk_size
            )

            loss = criterion(logits, labels)

            n = labels.size(0)

            total_loss += loss.item() * n
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += n

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.2%}"
            )

    if total_samples == 0:
        raise RuntimeError("Validation dataset is empty")

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "epoch",
        "stage",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_seconds",
        "peak_allocated_gb",
        "peak_reserved_gb"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


# --------------------------------------------------
# Main training
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/vision.yaml"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )
    parser.add_argument(
    "--smoke-test",
    action="store_true"
    )
    args = parser.parse_args()

    config_path = ROOT / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tc = config["training"]

    set_seed(tc["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. GPU training required."
        )

    device = torch.device("cuda")

    print("\nLSA64 ResNet-18 Local Training")
    print("--------------------------------")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"GPU VRAM: {gpu_memory:.2f} GB")

    # ----------------------------------------------
    # Datasets and loaders
    # ----------------------------------------------

    train_dataset = LSA64Dataset(
        split="train",
        num_frames=config["dataset"]["num_frames"]
    )

    val_dataset = LSA64Dataset(
        split="val",
        num_frames=config["dataset"]["num_frames"]
    )
    if args.smoke_test:
       from torch.utils.data import Subset

       train_dataset = Subset(train_dataset, range(8))
       val_dataset = Subset(val_dataset, range(4))

       print("SMOKE TEST: 8 training, 4 validation videos")

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        pin_memory=tc["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc["num_workers"],
        pin_memory=tc["pin_memory"]
    )

    print("Training videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))

    # ----------------------------------------------
    # Checkpoint paths
    # ----------------------------------------------

    checkpoint_dir = (
        ROOT / config["output"]["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = (
        ROOT / config["output"]["history_file"]
    )

    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    stages = [
        (
            "head",
            tc["head_epochs"],
            tc["head_learning_rate"]
        ),
        (
            "finetune",
            tc["finetune_epochs"],
            tc["finetune_learning_rate"]
        )
    ]

    # ----------------------------------------------
    # Resume handling
    # ----------------------------------------------

    resume_state = None

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)

        resume_state = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False
        )

    model = SignLanguageResNet(
        num_classes=config["dataset"]["num_classes"],
        pretrained=resume_state is None
    ).to(device)

    history = []
    best_accuracy = -1.0
    global_epoch = 0
    start_stage = 0
    start_epoch = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model"])

        history = resume_state["history"]
        best_accuracy = resume_state["best_accuracy"]
        global_epoch = resume_state["global_epoch"]

        start_stage = resume_state["stage_index"]
        start_epoch = resume_state["stage_epoch"] + 1
        
        resume_same_stage = True
        
        if start_epoch >= stages[start_stage][1]:
            start_stage += 1
            start_epoch = 0
            resume_same_stage = False

        if "rng_state" in resume_state:
            restore_rng_state(resume_state["rng_state"])

        print(f"Resuming after epoch {global_epoch}")

    accumulation = tc["gradient_accumulation_steps"]
    chunk_size = tc["frame_chunk_size"]
    use_amp = tc["mixed_precision"]

    if accumulation <= 0 or chunk_size <= 0:
        raise ValueError(
            "Accumulation and chunk size must be positive"
        )

    if tc["batch_size"] != 1:
        raise ValueError(
            "This training implementation requires batch_size=1"
        )

    # ----------------------------------------------
    # Two-stage training
    # ----------------------------------------------

    for stage_index, (stage, epochs, lr) in enumerate(stages):
        if stage_index < start_stage or epochs <=0:
            continue

        print(f"\nStarting stage: {stage.upper()}")

        configure_stage(model, stage)

        optimizer = torch.optim.AdamW(
            (
                p for p in model.parameters()
                if p.requires_grad
            ),
            lr=lr,
            weight_decay=tc["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1)
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp
        )

        first_epoch = (
            start_epoch if stage_index == start_stage
            else 0
        )

        patience_counter = 0
        stage_best = -1.0

        if(
            resume_state is not None
            and stage_index == start_stage
            and resume_same_stage
        ):
            optimizer.load_state_dict(
                resume_state["optimizer"]
            )
            scheduler.load_state_dict(
                resume_state["scheduler"]
            )
            scaler.load_state_dict(
                resume_state["scaler"]
            )

            patience_counter = resume_state.get(
                "patience_counter", 0
            )
            stage_best = resume_state.get(
                "stage_best", -1.0
            )

        for epoch in range(first_epoch, epochs):
            start_time = time.perf_counter()

            torch.cuda.reset_peak_memory_stats()

            model.train()
            freeze_batchnorm(model)

            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # --------------------------------------
            # Training progress bar
            # --------------------------------------

            train_iterator = iter(train_loader)

            progress = tqdm(
                total=len(train_loader),
                desc=f"Epoch {global_epoch + 1} [{stage}]",
                unit="video"
            )

            try:
                for group_start in range(
                    0, len(train_loader), accumulation
                ):
                    group_size = min(
                        accumulation,
                        len(train_loader) - group_start
                    )

                    for _ in range(group_size):
                        frames, labels = next(train_iterator)

                        frames = frames.to(
                            device, non_blocking=True
                        )

                        labels = labels.to(
                            device, non_blocking=True
                        )

                        loss, predictions = chunked_video_backward(
                            model,
                            frames,
                            labels,
                            chunk_size=chunk_size,
                            loss_scale=1.0 / group_size,
                            use_amp=use_amp,
                            scaler=scaler
                        )

                        n = labels.size(0)

                        total_loss += loss * n
                        total_correct += (
                            predictions == labels
                        ).sum().item()
                        total_samples += n

                        progress.update(n)

                        progress.set_postfix(
                            loss=f"{total_loss / total_samples:.4f}",
                            acc=f"{total_correct / total_samples:.2%}"
                        )

                    # Optimizer update after accumulation
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

            finally:
                progress.close()

            train_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples

            # --------------------------------------
            # Validation
            # --------------------------------------

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                device,
                chunk_size
            )

            scheduler.step()
            global_epoch += 1

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            peak_allocated = (
                torch.cuda.max_memory_allocated() / 1024**3
            )

            peak_reserved = (
                torch.cuda.max_memory_reserved() / 1024**3
            )

            print(
                f"\nEpoch {global_epoch} [{stage}] | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.2%} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_accuracy:.2%}"
            )

            print(
                f"Time: {elapsed:.1f}s | "
                f"GPU allocated: {peak_allocated:.3f} GB | "
                f"reserved: {peak_reserved:.3f} GB"
            )

            history.append({
                "epoch": global_epoch,
                "stage": stage,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_seconds": elapsed,
                "peak_allocated_gb": peak_allocated,
                "peak_reserved_gb": peak_reserved
            })

            # --------------------------------------
            # Best checkpoint and early stopping
            # --------------------------------------

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(
                    model.state_dict(),
                    best_path
                )

            if val_accuracy > stage_best:
                stage_best = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_accuracy": best_accuracy,
                "global_epoch": global_epoch,
                "stage_index": stage_index,
                "stage_epoch": epoch,
                "patience_counter": patience_counter,
                "stage_best": stage_best,
                "rng_state": capture_rng_state()
            }

            save_checkpoint(state, last_path)
            save_history(history, history_path)

            print("Checkpoint saved.")

            if patience_counter >= tc["patience"]:
                print(
                    f"Early stopping: no improvement for "
                    f"{patience_counter} epochs."
                )
                break

        resume_state = None
        start_epoch = 0

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
