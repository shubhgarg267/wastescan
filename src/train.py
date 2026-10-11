"""Training entry point. Pipeline (baseline_yolo / baseline_frcnn /
scale_fusion) is selected entirely by the `pipeline` key in the config
YAML — see experiments/configs/*.yaml. Baseline and contribution runs share
this same script and logging/checkpoint conventions so results are directly
comparable (see docs/contribution-notes.md).

Usage:
    python -m src.train --config experiments/configs/baseline_yolo.yaml
    python -m src.train --config experiments/configs/baseline_frcnn.yaml
    python -m src.train --config experiments/configs/scale_fusion.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.augmentation import build_train_augmentations
from src.data.dataset_loader import AerialWasteDataset, load_data_yaml
from src.data.tiled_dataset import TiledDetectionDataset
from src.models.baseline.faster_rcnn_baseline import (
    FasterRCNNBaseline,
    TorchvisionDetectionAdapter,
    collate_fn,
)
from src.models.baseline.yolo_baseline import YOLOv8Baseline
from src.models.scale_fusion.detector import ScaleFusionDetector


def setup_logging(log_file: str | Path) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def train_baseline_yolo(cfg: dict, logger: logging.Logger) -> None:
    data_yaml = Path(cfg["data"]["root"]) / "data.yaml"
    model = YOLOv8Baseline(
        model_variant=cfg["model"].get("yolo_variant", "yolov8n.pt"),
        device=cfg["train"].get("device"),
    )
    requested_epochs = cfg["train"]["epochs"]
    time_budget_minutes = cfg["train"].get("time_budget_minutes")
    time_budget_seconds = time_budget_minutes * 60 if time_budget_minutes else None

    logger.info(f"Training baseline_yolo on {data_yaml}")
    if time_budget_seconds is not None:
        logger.info(
            f"Soft time budget: {time_budget_minutes} min -- will stop after a completed "
            f"epoch (never mid-epoch) if another epoch would likely exceed it."
        )
    model.train(
        data_yaml=data_yaml,
        epochs=requested_epochs,
        imgsz=cfg["train"]["imgsz"],
        batch=cfg["train"]["batch_size"],
        project=cfg["output"]["project"],
        name=cfg["output"]["name"],
        time_budget_seconds=time_budget_seconds,
        logger=logger,
    )

    results_csv = Path(cfg["output"]["project"]).resolve() / cfg["output"]["name"] / "results.csv"
    completed_epochs = requested_epochs
    if results_csv.exists():
        with open(results_csv, "r") as f:
            completed_epochs = sum(1 for _ in f) - 1  # minus header row

    if completed_epochs < requested_epochs:
        logger.warning(
            f"Finished baseline_yolo training EARLY due to soft time budget: "
            f"{completed_epochs}/{requested_epochs} epochs completed."
        )
    else:
        logger.info(f"Finished baseline_yolo training: {completed_epochs}/{requested_epochs} epochs completed.")


def _run_torchvision_training(cfg: dict, logger: logging.Logger, model_wrapper, dataset) -> None:
    train_cfg = cfg["train"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.SGD(
        [p for p in model_wrapper.model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        momentum=train_cfg.get("momentum", 0.9),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    checkpoint_dir = Path(cfg["output"]["project"]) / cfg["output"]["name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        avg_loss = model_wrapper.train_one_epoch(
            loader, optimizer, log_every=train_cfg.get("log_every", 50)
        )
        logger.info(f"epoch {epoch + 1}/{train_cfg['epochs']} avg_loss={avg_loss:.4f}")
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        model_wrapper.save(ckpt_path)

    final_path = checkpoint_dir / "final.pt"
    model_wrapper.save(final_path)
    logger.info(f"Saved final checkpoint to {final_path}")


def train_baseline_frcnn(cfg: dict, logger: logging.Logger) -> None:
    ds_config = load_data_yaml(cfg["data"]["root"])
    transforms = build_train_augmentations(
        img_size=cfg["train"]["imgsz"],
        enable_occlusion_sim=cfg["data"].get("augmentation", {}).get("occlusion_sim", False),
    )
    base_dataset = AerialWasteDataset(cfg["data"]["root"], split="train", transforms=transforms)
    dataset = TorchvisionDetectionAdapter(base_dataset)

    model_wrapper = FasterRCNNBaseline(
        num_classes=ds_config.num_classes,
        pretrained=cfg["model"].get("pretrained_backbone", True),
        device=cfg["train"].get("device"),
    )
    logger.info(f"Training baseline_frcnn on {ds_config.root} ({ds_config.num_classes} classes)")
    _run_torchvision_training(cfg, logger, model_wrapper, dataset)


def train_scale_fusion(cfg: dict, logger: logging.Logger) -> None:
    ds_config = load_data_yaml(cfg["data"]["root"])
    base_dataset = AerialWasteDataset(cfg["data"]["root"], split="train", transforms=None)
    transforms = build_train_augmentations(
        img_size=cfg["train"]["imgsz"],
        enable_occlusion_sim=cfg["data"].get("augmentation", {}).get("occlusion_sim", True),
    )
    tiling_cfg = cfg["data"]["tiling"]
    dataset = TiledDetectionDataset(
        base_dataset,
        tile_size=tiling_cfg["tile_size"],
        overlap=tiling_cfg["overlap"],
        min_visibility=tiling_cfg["min_visibility"],
        transforms=transforms,
    )

    use_fusion = cfg["model"].get("use_fusion", True)
    model_wrapper = ScaleFusionDetector(
        num_classes=ds_config.num_classes,
        backbone_name=cfg["model"].get("backbone", "resnet50"),
        pretrained_backbone=cfg["model"].get("pretrained_backbone", True),
        use_fusion=use_fusion,
        device=cfg["train"].get("device"),
    )
    logger.info(
        f"Training scale_fusion on {ds_config.root} "
        f"({len(dataset)} tiles, {ds_config.num_classes} classes, use_fusion={use_fusion})"
    )
    _run_torchvision_training(cfg, logger, model_wrapper, dataset)


_PIPELINES = {
    "baseline_yolo": train_baseline_yolo,
    "baseline_frcnn": train_baseline_frcnn,
    "scale_fusion": train_scale_fusion,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg["output"]["log_file"])
    pipeline = cfg["pipeline"]
    if pipeline not in _PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Options: {list(_PIPELINES)}")

    logger.info(f"Config: {args.config}")
    _PIPELINES[pipeline](cfg, logger)


if __name__ == "__main__":
    main()
