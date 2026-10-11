"""Faster R-CNN (torchvision) baseline — second comparison detector, same
no-tiling/no-fusion/no-severity scope as the YOLOv8 baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.dataset_loader import AerialWasteDataset


class TorchvisionDetectionAdapter(Dataset):
    """Wraps AerialWasteDataset to the (image_tensor, target_dict) format
    torchvision detection models expect. Class ids are shifted +1 because
    torchvision reserves label 0 for background."""

    def __init__(self, base_dataset: AerialWasteDataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(sample["boxes_xyxy"]).float()
        labels = torch.from_numpy(sample["class_ids"]).long() + 1
        target = {
            "boxes": boxes if boxes.numel() else torch.zeros((0, 4), dtype=torch.float32),
            "labels": labels if labels.numel() else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_faster_rcnn(num_classes: int, pretrained: bool = True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 background
    return model


class FasterRCNNBaseline:
    def __init__(self, num_classes: int, pretrained: bool = True, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_faster_rcnn(num_classes, pretrained).to(self.device)

    def train_one_epoch(self, data_loader, optimizer, log_every: int = 50) -> float:
        self.model.train()
        running = []
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running.append(loss.item())
            if i % log_every == 0:
                print(f"  step {i}: loss={loss.item():.4f}")
        return sum(running) / max(1, len(running))

    @torch.no_grad()
    def predict(self, images: list[torch.Tensor], score_threshold: float = 0.25) -> list[dict]:
        self.model.eval()
        images = [img.to(self.device) for img in images]
        outputs = self.model(images)
        results = []
        for out in outputs:
            keep = out["scores"] >= score_threshold
            results.append(
                {
                    "boxes": out["boxes"][keep].cpu().numpy(),
                    "scores": out["scores"][keep].cpu().numpy(),
                    "labels": (out["labels"][keep] - 1).cpu().numpy(),
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
