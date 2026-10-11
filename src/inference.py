"""Full inference pipeline for a single aerial image: tiling (contribution
pipelines only) -> detection -> severity scoring -> annotated image +
severity report. Works with any of the three pipelines via the same config
files used by train.py/evaluate.py.

Programmatic use (e.g. app.py): call load_model() once and reuse it across
requests with run_inference_with_model() -- loading the model is the
expensive part. run_inference() is a one-shot convenience wrapper (loads
fresh each call) used by the CLI below.

Usage:
    python -m src.inference --config experiments/configs/baseline_yolo.yaml --image path/to/site.jpg
    python -m src.inference --config experiments/configs/scale_fusion.yaml --weights experiments/checkpoints/scale_fusion/final.pt --image path/to/site.jpg
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.dataset_loader import load_data_yaml
from src.data.tiling import merge_tile_detections, tile_image
from src.models.baseline.faster_rcnn_baseline import FasterRCNNBaseline
from src.models.baseline.yolo_baseline import YOLOv8Baseline
from src.models.scale_fusion.detector import ScaleFusionDetector
from src.models.severity_scoring import SeverityReport, compute_severity_report, draw_severity_overlay

ImageInput = "str | Path | Image.Image | np.ndarray"


def load_image_bgr_and_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_bgr_and_rgb(image) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes a path, PIL Image, or RGB numpy array into (bgr, rgb) arrays."""
    if isinstance(image, (str, Path)):
        return load_image_bgr_and_rgb(image)
    if isinstance(image, np.ndarray):
        rgb = image
    else:  # PIL Image
        rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, rgb


def run_yolo_inference(model: YOLOv8Baseline, source, imgsz: int, conf: float):
    results = model.predict(source=source, imgsz=imgsz, conf=conf)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    class_names = [r.names[i] for i in sorted(r.names)]
    return boxes, scores, class_ids, class_names


@torch.no_grad()
def run_torchvision_inference(model, image_rgb: np.ndarray, conf: float):
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    out = model.predict([tensor], score_threshold=conf)[0]
    return out["boxes"], out["scores"], out["labels"]


@torch.no_grad()
def run_tiled_inference(
    model, image_rgb: np.ndarray, tile_size: int, overlap: float, conf: float, iou_threshold: float = 0.5
):
    tiles = tile_image(image_rgb, tile_size, overlap)
    tile_coords = [c for _, c in tiles]
    patches = [
        torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1).float() / 255.0 for p, _ in tiles
    ]
    outputs = model.predict(patches, score_threshold=conf)
    merged = merge_tile_detections(
        [o["boxes"] for o in outputs],
        [o["scores"] for o in outputs],
        [o["labels"] for o in outputs],
        tile_coords,
        iou_threshold=iou_threshold,
    )
    return merged["boxes"], merged["scores"], merged["class_ids"]


@dataclasses.dataclass
class LoadedModel:
    cfg: dict
    pipeline: str
    model: object
    class_names: list[str] | None


def load_model(config_path: str | Path, weights: str | None = None) -> LoadedModel:
    """Loads config + the appropriate model for its pipeline -- the expensive
    step; callers that serve repeated requests (e.g. a web UI) should call
    this once and cache the result rather than calling it per-request."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    pipeline = cfg["pipeline"]

    if pipeline == "baseline_yolo":
        model = (
            YOLOv8Baseline.from_checkpoint(weights)
            if weights
            else YOLOv8Baseline(model_variant=cfg["model"].get("yolo_variant", "yolov8n.pt"))
        )
        class_names = None  # ultralytics resolves its own names dict at predict time
    elif pipeline == "baseline_frcnn":
        ds_config = load_data_yaml(cfg["data"]["root"])
        class_names = ds_config.class_names
        model = FasterRCNNBaseline(
            num_classes=ds_config.num_classes, pretrained=False, device=cfg["train"].get("device")
        )
        if weights:
            model.load(weights)
    elif pipeline == "scale_fusion":
        ds_config = load_data_yaml(cfg["data"]["root"])
        class_names = ds_config.class_names
        model = ScaleFusionDetector(
            num_classes=ds_config.num_classes,
            backbone_name=cfg["model"].get("backbone", "resnet50"),
            pretrained_backbone=False,
            use_fusion=cfg["model"].get("use_fusion", True),
            device=cfg["train"].get("device"),
        )
        if weights:
            model.load(weights)
    else:
        raise ValueError(f"Unknown pipeline '{pipeline}'")

    return LoadedModel(cfg=cfg, pipeline=pipeline, model=model, class_names=class_names)


def _detect(loaded: LoadedModel, image_rgb: np.ndarray, image_path: Path | None, conf: float):
    cfg, pipeline, model = loaded.cfg, loaded.pipeline, loaded.model
    if pipeline == "baseline_yolo":
        yolo_source = str(image_path) if image_path is not None else Image.fromarray(image_rgb)
        boxes, scores, class_ids, resolved_names = run_yolo_inference(
            model, yolo_source, cfg["train"]["imgsz"], conf
        )
        return boxes, scores, class_ids, resolved_names
    if pipeline == "baseline_frcnn":
        boxes, scores, class_ids = run_torchvision_inference(model, image_rgb, conf)
        return boxes, scores, class_ids, None
    if pipeline == "scale_fusion":
        tiling_cfg = cfg["data"]["tiling"]
        boxes, scores, class_ids = run_tiled_inference(
            model, image_rgb, tiling_cfg["tile_size"], tiling_cfg["overlap"], conf
        )
        return boxes, scores, class_ids, None
    raise ValueError(f"Unknown pipeline '{pipeline}'")


def run_inference_with_model(
    loaded: LoadedModel,
    image,
    conf_threshold: float = 0.25,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> tuple[np.ndarray, list[dict], SeverityReport]:
    """Runs tiling -> detection -> severity scoring on one image using an
    already-loaded model. Returns (annotated_rgb_image, detections, report)
    without touching disk. `image` accepts a file path, PIL Image, or RGB
    numpy array."""
    cfg = loaded.cfg
    severity_cfg = cfg.get("severity", {})
    grid_rows = grid_rows or severity_cfg.get("grid_rows", 3)
    grid_cols = grid_cols or severity_cfg.get("grid_cols", 3)
    level_thresholds = severity_cfg.get("level_thresholds")

    image_path = image if isinstance(image, (str, Path)) else None
    image_bgr, image_rgb = to_bgr_and_rgb(image)

    boxes, scores, class_ids, resolved_class_names = _detect(loaded, image_rgb, image_path, conf_threshold)
    class_names = resolved_class_names or loaded.class_names or ["waste"]

    img_h, img_w = image_rgb.shape[:2]
    report_kwargs = {"level_thresholds": level_thresholds} if level_thresholds else {}
    report = compute_severity_report(
        img_h,
        img_w,
        boxes,
        class_ids,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        image_path=str(image_path) if image_path else None,
        **report_kwargs,
    )

    annotated_bgr = draw_severity_overlay(
        image_bgr, report, boxes_xyxy=boxes, class_ids=class_ids, class_names=class_names
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = [
        {
            "box_xyxy": [float(v) for v in box],
            "score": float(score),
            "class_id": int(cls_id),
            "class_name": (
                class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else str(cls_id)
            ),
        }
        for box, score, cls_id in zip(boxes, scores, class_ids)
    ]

    return annotated_rgb, detections, report


def run_inference(
    config_path: str | Path,
    image,
    conf_threshold: float = 0.25,
    weights: str | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> tuple[np.ndarray, list[dict], SeverityReport]:
    """One-shot convenience wrapper: loads the model fresh, then runs
    inference. For repeated calls, use load_model() + run_inference_with_model()
    instead so the model is only loaded once."""
    loaded = load_model(config_path, weights)
    return run_inference_with_model(loaded, image, conf_threshold, grid_rows, grid_cols)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output-dir", default="results/qualitative", type=str)
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_rgb, detections, report = run_inference(
        args.config,
        image_path,
        conf_threshold=args.conf,
        weights=args.weights,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    report_path = output_dir / f"{stem}_severity_report.json"
    cv2.imwrite(str(annotated_path), annotated_bgr)
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Detections: {len(detections)}")
    print(f"Overall severity: {report.overall_level} (score={report.overall_score:.3f})")
    print(f"Annotated image: {annotated_path}")
    print(f"Severity report: {report_path}")


if __name__ == "__main__":
    main()
