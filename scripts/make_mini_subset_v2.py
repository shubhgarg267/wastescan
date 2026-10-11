"""Creates a STRATIFIED subset of the YOLO dataset -- same copy-based
structure as make_mini_subset.py, but stratifies each split by annotation
presence (has-boxes vs background-only image) instead of sampling purely
at random. KACHARA is single-class (nc=1), so "has at least one waste box"
vs "background-only tile" is the only meaningful stratum available; this
keeps the subset's empty/non-empty ratio matching the source split's ratio
rather than leaving it to chance, which plain random sampling
(make_mini_subset.py) doesn't guarantee.

Usage:
    python scripts/make_mini_subset_v2.py --root data/raw --out data/processed/mini_v2 \
        --train-count 2000 --valid-count 400 --seed 0
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


def _has_annotation(label_path: Path) -> bool:
    return label_path.exists() and label_path.stat().st_size > 0


def sample_split_stratified(
    root: Path, split_dir: str, out_root: Path, out_split: str, count: int, seed: int
) -> dict:
    images_dir = root / split_dir / "images"
    labels_dir = root / split_dir / "labels"
    image_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    positives, negatives = [], []
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        (positives if _has_annotation(label_path) else negatives).append(img_path)

    rng = random.Random(seed)
    pos_frac = len(positives) / len(image_paths) if image_paths else 0.0
    count = min(count, len(image_paths))

    n_pos = min(round(count * pos_frac), len(positives))
    n_neg = min(count - n_pos, len(negatives))
    shortfall = count - (n_pos + n_neg)
    if shortfall > 0:  # top up from whichever stratum still has slack
        if len(positives) - n_pos >= len(negatives) - n_neg:
            n_pos += min(shortfall, len(positives) - n_pos)
        else:
            n_neg += min(shortfall, len(negatives) - n_neg)

    sampled = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(sampled)

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

    return {
        "total": len(sampled),
        "with_annotations": n_pos,
        "background_only": n_neg,
        "source_pos_frac": pos_frac,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", type=str)
    parser.add_argument("--out", default="data/processed/mini_v2", type=str)
    parser.add_argument("--train-count", default=2000, type=int)
    parser.add_argument("--valid-count", default=400, type=int)
    parser.add_argument("--test-count", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    config = load_data_yaml(root)

    requested = {"train": args.train_count, "valid": args.valid_count}
    if args.test_count:
        requested["test"] = args.test_count

    for logical_split, count in requested.items():
        if logical_split not in config.split_dirs:
            print(f"WARNING: split '{logical_split}' not found in {root}, skipping")
            continue
        stats = sample_split_stratified(
            root, config.split_dirs[logical_split], out_root, logical_split, count, args.seed
        )
        print(
            f"{logical_split}: sampled {stats['total']} images "
            f"({stats['with_annotations']} with annotations, "
            f"{stats['background_only']} background-only, "
            f"source pos-frac={stats['source_pos_frac']:.3f}) -> {out_root / logical_split}"
        )

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        if args.test_count:
            f.write("test: test/images\n")
        f.write(f"nc: {config.num_classes}\n")
        f.write(f"names: {config.class_names}\n")

    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    main()
