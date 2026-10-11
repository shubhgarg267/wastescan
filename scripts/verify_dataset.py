"""Sanity-checks a YOLO-format dataset export before training: confirms
data.yaml + train/valid/test layout, reports image/label counts and class
distribution. Run this first after dropping KACHARA files into data/raw/.

Usage:
    python scripts/verify_dataset.py --root data/raw
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import AerialWasteDataset, load_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", type=str)
    args = parser.parse_args()

    config = load_data_yaml(args.root)
    print(f"Dataset root: {config.root}")
    print(f"Classes ({config.num_classes}): {config.class_names}")
    print(f"Splits found: {list(config.split_dirs)}")

    # "val" and "valid" may both map to the same on-disk folder (dataset_loader
    # accepts either name) -- report each physical split once.
    seen_dirs = set()
    unique_splits = []
    for split, split_dir in config.split_dirs.items():
        if split_dir not in seen_dirs:
            seen_dirs.add(split_dir)
            unique_splits.append(split)

    for split in unique_splits:
        dataset = AerialWasteDataset(args.root, split=split)
        class_counts = Counter()
        empty_labels = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["class_ids"].size == 0:
                empty_labels += 1
            for cid in sample["class_ids"]:
                class_counts[config.class_names[int(cid)]] += 1
        print(f"\n[{split}] {len(dataset)} images, {empty_labels} with no annotations")
        for name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
