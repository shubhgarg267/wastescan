"""Draws detections + zone severity overlay for qualitative output figures
and inference.py's annotated-image output."""

from __future__ import annotations

import cv2
import numpy as np

from src.models.severity_scoring.scoring import SeverityReport

_LEVEL_COLORS = {
    "low": (0, 200, 0),      # BGR: green
    "medium": (0, 165, 255), # orange
    "high": (0, 0, 255),     # red
}


def draw_severity_overlay(
    image: np.ndarray,
    report: SeverityReport,
    boxes_xyxy: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    class_names: list[str] | None = None,
    alpha: float = 0.25,
) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()

    for zone in report.zones:
        x1, y1, x2, y2 = zone.bbox
        color = _LEVEL_COLORS[zone.severity_level]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)
        label = f"{zone.severity_level} ({zone.coverage_fraction:.2f})"
        cv2.putText(
            canvas, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    if boxes_xyxy is not None:
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
            if class_ids is not None and class_names is not None:
                name = class_names[int(class_ids[i])]
                cv2.putText(
                    canvas, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )

    banner = f"Overall severity: {report.overall_level.upper()} (score={report.overall_score:.2f}, {report.num_detections} detections)"
    cv2.putText(canvas, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return canvas
