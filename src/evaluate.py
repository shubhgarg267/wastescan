"""Evaluation entry point — same config-flag pipeline switch as train.py.
Logs mAP (via pycocotools) and IoU@0.5 precision/recall to
experiments/logs/. For scale_fusion, full images are tiled, predicted on
per-tile, then reprojected + merged (src/data/tiling.py) before scoring
against full-image ground truth, so metrics are directly comparable to the
no-tiling baselines.

Usage:
    python -m src.evaluate --config experiments/configs/baseline_yolo.yaml --split test
    python -m src.evaluate --config experiments/configs/baseline_frcnn.yaml --split test --weights experiments/checkpoints/baseline_frcnn/final.pt
    python -m src.evaluate --config experiments/configs/scale_fusion.yaml --split test --weights experiments/checkpoints/scale_fusion/final.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.dataset_loader import AerialWasteDataset, load_data_yaml
from src.data.tiling import merge_tile_detections, tile_image
from src.models.baseline.faster_rcnn_baseline import FasterRCNNBaseline
from src.models.baseline.yolo_baseline import YOLOv8Baseline
from src.models.scale_fusion.detector import ScaleFusionDetector
from src.train import setup_logging


def build_coco_gt(dataset: AerialWasteDataset) -> dict:
    images, annotations, categories = [], [], []
    for cid, name in enumerate(dataset.class_names):
        categories.append({"id": cid, "name": name, "supercategory": "waste"})

    ann_id = 1
    for idx in range(len(dataset)):
        sample = dataset[idx]
        h, w = sample["orig_size"]
        images.append({"id": idx, "file_name": sample["image_path"], "height": h, "width": w})
        for box, cls in zip(sample["boxes_xyxy"], sample["class_ids"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def _iou_single(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def compute_precision_recall(
    coco_gt_dict: dict, predictions: list[dict], iou_threshold: float = 0.5, score_threshold: float = 0.25
) -> dict:
    gt_by_img_cls = defaultdict(list)
    for ann in coco_gt_dict["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_img_cls[(ann["image_id"], ann["category_id"])].append([x, y, x + w, y + h])

    matched = defaultdict(set)
    tp, fp = 0, 0
    preds = sorted((p for p in predictions if p["score"] >= score_threshold), key=lambda p: -p["score"])
    for p in preds:
        key = (p["image_id"], p["category_id"])
        gts = gt_by_img_cls.get(key, [])
        x, y, w, h = p["bbox"]
        pred_box = [x, y, x + w, y + h]
        best_iou, best_idx = 0.0, -1
        for i, gt_box in enumerate(gts):
            if i in matched[key]:
                continue
            iou = _iou_single(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            matched[key].add(best_idx)
            tp += 1
        else:
            fp += 1

    num_gt = sum(len(v) for v in gt_by_img_cls.values())
    fn = num_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "num_gt": num_gt}


def run_coco_eval(coco_gt_dict: dict, predictions: list[dict]) -> dict:
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    if not predictions:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "AR_100": 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP_50_95": coco_eval.stats[0], "mAP_50": coco_eval.stats[1], "AR_100": coco_eval.stats[8]}


def evaluate_baseline_yolo(cfg: dict, split: str, weights: str | None) -> dict:
    data_yaml = Path(cfg["data"]["root"]) / "data.yaml"
    model = (
        YOLOv8Baseline.from_checkpoint(weights)
        if weights
        else YOLOv8Baseline(model_variant=cfg["model"].get("yolo_variant", "yolov8n.pt"))
    )
    results = model.evaluate(data_yaml=data_yaml, split=split, imgsz=cfg["train"]["imgsz"])
    return {
        "mAP_50_95": float(results.box.map),
        "mAP_50": float(results.box.map50),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


@torch.no_grad()
def evaluate_baseline_frcnn(cfg: dict, split: str, weights: str | None) -> dict:
    ds_config = load_data_yaml(cfg["data"]["root"])
    dataset = AerialWasteDataset(cfg["data"]["root"], split=split, transforms=None)
    model = FasterRCNNBaseline(
        num_classes=ds_config.num_classes,
        pretrained=False,
        device=cfg["train"].get("device"),
    )
    if weights:
        model.load(weights)

    coco_gt_dict = build_coco_gt(dataset)
    predictions = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_tensor = torch.from_numpy(sample["image"]).permute(2, 0, 1).float() / 255.0
        out = model.predict([image_tensor], score_threshold=0.001)[0]
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                {
                    "image_id": idx,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    metrics = run_coco_eval(coco_gt_dict, predictions)
    metrics.update(compute_precision_recall(coco_gt_dict, predictions))
    return metrics


@torch.no_grad()
def evaluate_scale_fusion(cfg: dict, split: str, weights: str | None) -> dict:
    ds_config = load_data_yaml(cfg["data"]["root"])
    dataset = AerialWasteDataset(cfg["data"]["root"], split=split, transforms=None)
    model = ScaleFusionDetector(
        num_classes=ds_config.num_classes,
        backbone_name=cfg["model"].get("backbone", "resnet50"),
        pretrained_backbone=False,
        use_fusion=cfg["model"].get("use_fusion", True),
        device=cfg["train"].get("device"),
    )
    if weights:
        model.load(weights)

    tiling_cfg = cfg["data"]["tiling"]
    coco_gt_dict = build_coco_gt(dataset)
    predictions = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        tiles = tile_image(sample["image"], tiling_cfg["tile_size"], tiling_cfg["overlap"])
        patches = [
            torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1).float() / 255.0 for p, _ in tiles
        ]
        tile_coords = [c for _, c in tiles]
        outputs = model.predict(patches, score_threshold=0.001)
        merged = merge_tile_detections(
            [o["boxes"] for o in outputs],
            [o["scores"] for o in outputs],
            [o["labels"] for o in outputs],
            tile_coords,
            iou_threshold=0.5,
        )
        for box, score, label in zip(merged["boxes"], merged["scores"], merged["class_ids"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                {
                    "image_id": idx,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    metrics = run_coco_eval(coco_gt_dict, predictions)
    metrics.update(compute_precision_recall(coco_gt_dict, predictions))
    return metrics


_PIPELINES = {
    "baseline_yolo": evaluate_baseline_yolo,
    "baseline_frcnn": evaluate_baseline_frcnn,
    "scale_fusion": evaluate_scale_fusion,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--weights", default=None, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg["output"]["log_file"])
    pipeline = cfg["pipeline"]
    if pipeline not in _PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Options: {list(_PIPELINES)}")

    logger.info(f"Evaluating {pipeline} on split={args.split} (weights={args.weights})")
    metrics = _PIPELINES[pipeline](cfg, args.split, args.weights)
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    results_path = Path("results/tables") / f"{cfg['output']['name']}_{args.split}_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {results_path}")


if __name__ == "__main__":
    main()
