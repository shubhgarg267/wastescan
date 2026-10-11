"""Creates a small random subset of the YOLO dataset for fast CPU
iteration — copies (not symlinks, for Windows compatibility) a capped
number of image+label pairs per split into a new root with its own
data.yaml, reusing the source class list.

Usage:
    python scripts/make_mini_subset.py --root data/raw --out data/processed/mini \
        --train-count 900 --valid-count 180 --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import load_data_yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def sample_split(
    root: Path, split_dir: str, out_root: Path, out_split: str, count: int, seed: int
) -> int:
    images_dir = root / split_dir / "images"
    labels_dir = root / split_dir / "labels"
    image_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    rng = random.Random(seed)
    count = min(count, len(image_paths))
    sampled = rng.sample(image_paths, count)

    out_images = out_root / out_split / "images"
    out_labels = out_root / out_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path in sampled:
        shutil.copy2(img_path, out_images / img_path.name)
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_labels / label_path.name)
        else:
            (out_labels / f"{img_path.stem}.txt").touch()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", type=str)
    parser.add_argument("--out", default="data/processed/mini", type=str)
    parser.add_argument("--train-count", default=900, type=int)
    parser.add_argument("--valid-count", default=180, type=int)
    parser.add_argument("--test-count", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    config = load_data_yaml(root)

    requested = {"train": args.train_count, "valid": args.valid_count}
    if args.test_count:
        requested["test"] = args.test_count

    actual_counts = {}
    for logical_split, count in requested.items():
        if logical_split not in config.split_dirs:
            print(f"WARNING: split '{logical_split}' not found in {root}, skipping")
            continue
        actual = sample_split(
            root, config.split_dirs[logical_split], out_root, logical_split, count, args.seed
        )
        actual_counts[logical_split] = actual
        print(f"{logical_split}: sampled {actual} images -> {out_root / logical_split}")

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        if "test" in actual_counts:
            f.write("test: test/images\n")
        f.write(f"nc: {config.num_classes}\n")
        f.write(f"names: {config.class_names}\n")

    print(f"\nWrote {yaml_path}")
    print(f"Total images: {sum(actual_counts.values())}")


if __name__ == "__main__":
    main()
