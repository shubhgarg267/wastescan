"""Multi-scale feature fusion block — contribution 1 (see
docs/contribution-notes.md). Takes the per-level feature maps a standard
FPN already produces and additionally lets every level see a weighted,
learned summary of *all* levels before the detection head runs, so a small
or partially-occluded item whose only strong signal sits in one
high-resolution level still benefits from context found at other scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFusionBlock(nn.Module):
    def __init__(self, in_channels: int, levels: list[str], fused_channels: int | None = None):
        super().__init__()
        self.levels = levels
        fused_channels = fused_channels or in_channels
        self.level_weights = nn.Parameter(torch.ones(len(levels)))
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )
        self.inject_convs = nn.ModuleDict(
            {level: nn.Conv2d(fused_channels, in_channels, kernel_size=1) for level in levels}
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_size = features[self.levels[0]].shape[-2:]
        weights = F.softmax(self.level_weights, dim=0)

        resized = []
        for w, level in zip(weights, self.levels):
            feat = features[level]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            resized.append(w * feat)
        fused = self.fuse_conv(torch.stack(resized, dim=0).sum(dim=0))

        out = {}
        for level in self.levels:
            feat = features[level]
            level_size = feat.shape[-2:]
            fused_resized = (
                fused
                if level_size == target_size
                else F.interpolate(fused, size=level_size, mode="bilinear", align_corners=False)
            )
            out[level] = feat + self.inject_convs[level](fused_resized)
        return out
