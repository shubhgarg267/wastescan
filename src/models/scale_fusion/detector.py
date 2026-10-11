"""Scale-fusion detector: standard ResNet-FPN backbone + ScaleFusionBlock
inserted before the RPN/ROI heads, wrapped in torchvision's Faster R-CNN
detection framework. This is the model half of contribution 1 — paired
with the tiling in src/data/tiling.py (see docs/contribution-notes.md).
Same train/predict interface as the baselines so train.py/evaluate.py can
switch between them via config flag alone.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from src.models.scale_fusion.fusion_head import ScaleFusionBlock

FPN_LEVELS = ["0", "1", "2", "3", "pool"]


class ScaleFusionBackbone(nn.Module):
    def __init__(self, base_backbone: nn.Module, fusion_block: ScaleFusionBlock):
        super().__init__()
        self.base_backbone = base_backbone
        self.fusion = fusion_block
        self.out_channels = base_backbone.out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.base_backbone(x)
        return self.fusion(features)


def build_scale_fusion_faster_rcnn(
    num_classes: int,
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = True,
    use_fusion: bool = True,
) -> FasterRCNN:
    base_backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights="DEFAULT" if pretrained_backbone else None,
    )
    if not use_fusion:
        # tiling-only ablation: plain FPN backbone, no ScaleFusionBlock -- still
        # trained/evaluated through the same tiled pipeline (src/data/tiling.py)
        return FasterRCNN(base_backbone, num_classes=num_classes + 1)  # +1 background
    fusion_block = ScaleFusionBlock(in_channels=base_backbone.out_channels, levels=FPN_LEVELS)
    backbone = ScaleFusionBackbone(base_backbone, fusion_block)
    return FasterRCNN(backbone, num_classes=num_classes + 1)  # +1 background


class ScaleFusionDetector:
    """Train/predict wrapper mirroring FasterRCNNBaseline's interface.

    use_fusion=False builds the "tiling-only" ablation variant (plain FPN
    backbone, no ScaleFusionBlock) instead of the full fusion detector --
    both go through the identical tiled train/eval pipeline, so this flag is
    the only difference between the two ablation legs.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        use_fusion: bool = True,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fusion = use_fusion
        self.model = build_scale_fusion_faster_rcnn(
            num_classes, backbone_name, pretrained_backbone, use_fusion=use_fusion
        ).to(self.device)

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
