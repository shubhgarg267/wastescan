"""Pre-generates overlapping tiles + YOLO-format labels for a dataset split
into data/processed/ — useful for inspecting tiles by eye, or for training
setups that prefer reading tiles from disk over tiling on the fly (the
default TiledDetectionDataset in src/data/tiled_dataset.py tiles on the fly
and does not require this script to be run).

Usage:
    python scripts/tile_dataset.py --root data/raw --split train --out data/processed/train --tile-size 640 --overlap 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import AerialWasteDataset, xyxy_to_yolo_norm
from src.data.tiling import clip_boxes_to_tile, compute_tile_coords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--tile-size", default=640, type=int)
    parser.add_argument("--overlap", default=0.2, type=float)
    parser.add_argument("--min-visibility", default=0.3, type=float)
    args = parser.parse_args()

    dataset = AerialWasteDataset(args.root, split=args.split, transforms=None)
    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        img_h, img_w = image.shape[:2]
        stem = Path(sample["image_path"]).stem

        coords = compute_tile_coords(img_w, img_h, args.tile_size, args.overlap)
        for coord in coords:
            patch = image[
                coord.y_offset : coord.y_offset + coord.height,
                coord.x_offset : coord.x_offset + coord.width,
            ]
            boxes, classes = clip_boxes_to_tile(
                sample["boxes_xyxy"], sample["class_ids"], coord, args.min_visibility
            )
            tile_name = f"{stem}_tile{coord.tile_index}"
            cv2.imwrite(
                str(out_root / "images" / f"{tile_name}.jpg"),
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
            )
            boxes_norm = xyxy_to_yolo_norm(boxes, coord.width, coord.height)
            with open(out_root / "labels" / f"{tile_name}.txt", "w") as f:
                for box, cls in zip(boxes_norm, classes):
                    f.write(f"{int(cls)} {' '.join(f'{v:.6f}' for v in box)}\n")
            total_tiles += 1

    print(f"Wrote {total_tiles} tiles from {len(dataset)} images to {out_root}")


if __name__ == "__main__":
    main()
