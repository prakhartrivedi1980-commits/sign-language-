
from pathlib import Path
from collections import Counter
import re
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config_path = PROJECT_ROOT / "configs" / "split.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset_root = PROJECT_ROOT / config["dataset"]["root"]

train_signers = set(config["split"]["train_signers"])
val_signers = set(config["split"]["val_signers"])
test_signers = set(config["split"]["test_signers"])

# Ensure no signer belongs to multiple splits
assert train_signers.isdisjoint(val_signers)
assert train_signers.isdisjoint(test_signers)
assert val_signers.isdisjoint(test_signers)

assert train_signers | val_signers | test_signers == set(range(1, 11))

counts = Counter()
classes = {
    "train": set(),
    "val": set(),
    "test": set()
}
seen = set()

pattern = re.compile(r"^(\d{3})_(\d{3})_(\d{3})\.mp4$")

videos = sorted(dataset_root.glob("*.mp4"))

for video in videos:
    match = pattern.fullmatch(video.name)
    assert match, f"Invalid filename: {video.name}"

    class_id, signer_id, repetition = map(int, match.groups())

    assert 1 <= class_id <= 64
    assert 1 <= signer_id <= 10
    assert 1 <= repetition <= 5

    key = (class_id, signer_id, repetition)
    assert key not in seen, f"Duplicate: {video.name}"
    seen.add(key)

    if signer_id in train_signers:
        split = "train"
    elif signer_id in val_signers:
        split = "val"
    elif signer_id in test_signers:
        split = "test"
    else:
        raise ValueError(f"Unknown signer: {signer_id}")

    counts[split] += 1
    classes[split].add(class_id)

print("\nLSA64 Dataset Split Verification")
print("--------------------------------")
print("Training videos:", counts["train"])
print("Validation videos:", counts["val"])
print("Testing videos:", counts["test"])
print("Total videos:", len(videos))

assert len(videos) == 3200
assert len(seen) == 3200
assert counts["train"] == 2240
assert counts["val"] == 320
assert counts["test"] == 640

for split in ("train", "val", "test"):
    assert len(classes[split]) == 64

print("\nAll signer-independent split checks passed!")