"""Dumping severity scoring — contribution 2, the genuinely new part (see
docs/contribution-notes.md). Existing aerial-waste work, including KACHARA,
stops at detection/classification ("waste present/absent"). This module
takes detections for a site and turns them into a per-zone coverage/density
read and an overall severity score, so cleanup can be prioritized instead
of just flagged.
"""

from __future__ import annotations

import dataclasses

import numpy as np

DEFAULT_LEVEL_THRESHOLDS = {"low": 0.10, "medium": 0.30}  # coverage-fraction cutoffs, tune per site


@dataclasses.dataclass
class ZoneScore:
    zone_index: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in full-image pixel coords
    coverage_fraction: float
    num_detections: int
    severity_level: str


@dataclasses.dataclass
class SeverityReport:
    image_path: str | None
    image_size: tuple[int, int]  # (h, w)
    grid_shape: tuple[int, int]  # (rows, cols)
    zones: list[ZoneScore]
    num_detections: int
    overall_coverage: float
    overall_score: float
    overall_level: str

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": self.image_size,
            "grid_shape": self.grid_shape,
            "num_detections": self.num_detections,
            "overall_coverage": round(self.overall_coverage, 4),
            "overall_score": round(self.overall_score, 4),
            "overall_level": self.overall_level,
            "zones": [
                {
                    "zone_index": z.zone_index,
                    "row": z.row,
                    "col": z.col,
                    "bbox": z.bbox,
                    "coverage_fraction": round(z.coverage_fraction, 4),
                    "num_detections": z.num_detections,
                    "severity_level": z.severity_level,
                }
                for z in self.zones
            ],
        }


def level_from_coverage(coverage: float, thresholds: dict[str, float] = DEFAULT_LEVEL_THRESHOLDS) -> str:
    if coverage < thresholds["low"]:
        return "low"
    if coverage < thresholds["medium"]:
        return "medium"
    return "high"


def _coverage_mask(img_h: int, img_w: int, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Rasterizes boxes onto a boolean canvas so overlapping/piled items are
    counted once (union), not summed — a plain area-sum would overstate
    coverage wherever waste is stacked, which is exactly the occlusion case
    this project cares about."""
    mask = np.zeros((img_h, img_w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
        xi2, yi2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
    return mask


def compute_severity_report(
    img_h: int,
    img_w: int,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray | None = None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    image_path: str | None = None,
    level_thresholds: dict[str, float] = DEFAULT_LEVEL_THRESHOLDS,
) -> SeverityReport:
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    mask = _coverage_mask(img_h, img_w, boxes_xyxy)

    centers = None
    if boxes_xyxy.size:
        centers = np.stack(
            [(boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2, (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2],
            axis=1,
        )

    row_edges = np.linspace(0, img_h, grid_rows + 1).astype(int)
    col_edges = np.linspace(0, img_w, grid_cols + 1).astype(int)

    zones = []
    zone_idx = 0
    for r in range(grid_rows):
        y1, y2 = row_edges[r], row_edges[r + 1]
        for c in range(grid_cols):
            x1, x2 = col_edges[c], col_edges[c + 1]
            zone_mask = mask[y1:y2, x1:x2]
            coverage = float(zone_mask.mean()) if zone_mask.size else 0.0

            num_dets = 0
            if centers is not None:
                in_zone = (
                    (centers[:, 0] >= x1) & (centers[:, 0] < x2)
                    & (centers[:, 1] >= y1) & (centers[:, 1] < y2)
                )
                num_dets = int(in_zone.sum())

            zones.append(
                ZoneScore(
                    zone_index=zone_idx,
                    row=r,
                    col=c,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    coverage_fraction=coverage,
                    num_detections=num_dets,
                    severity_level=level_from_coverage(coverage, level_thresholds),
                )
            )
            zone_idx += 1

    overall_coverage = float(mask.mean())
    overall_score = float(np.mean([z.coverage_fraction for z in zones])) if zones else 0.0

    return SeverityReport(
        image_path=image_path,
        image_size=(img_h, img_w),
        grid_shape=(grid_rows, grid_cols),
        zones=zones,
        num_detections=int(boxes_xyxy.shape[0]),
        overall_coverage=overall_coverage,
        overall_score=overall_score,
        overall_level=level_from_coverage(overall_score, level_thresholds),
    )


def rank_sites(reports: list[SeverityReport]) -> list[SeverityReport]:
    """Sorts sites by overall_score descending — highest-priority cleanup first."""
    return sorted(reports, key=lambda r: r.overall_score, reverse=True)
