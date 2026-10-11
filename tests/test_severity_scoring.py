import numpy as np

from src.models.severity_scoring.scoring import (
    compute_severity_report,
    level_from_coverage,
    rank_sites,
)


def test_level_from_coverage_thresholds():
    assert level_from_coverage(0.05) == "low"
    assert level_from_coverage(0.15) == "medium"
    assert level_from_coverage(0.50) == "high"


def test_compute_severity_report_no_detections_is_all_low():
    report = compute_severity_report(
        img_h=300, img_w=300, boxes_xyxy=np.zeros((0, 4)), class_ids=np.zeros((0,)), grid_rows=3, grid_cols=3
    )
    assert report.num_detections == 0
    assert report.overall_level == "low"
    assert len(report.zones) == 9
    assert all(z.coverage_fraction == 0.0 for z in report.zones)


def test_compute_severity_report_single_zone_fully_covered():
    # A 300x300 image split 3x3 -> each zone is 100x100. Fill zone (0,0) entirely.
    boxes = np.array([[0.0, 0.0, 100.0, 100.0]])
    classes = np.array([0])
    report = compute_severity_report(300, 300, boxes, classes, grid_rows=3, grid_cols=3)

    zone00 = next(z for z in report.zones if z.row == 0 and z.col == 0)
    assert zone00.coverage_fraction == 1.0
    assert zone00.severity_level == "high"

    other_zones = [z for z in report.zones if not (z.row == 0 and z.col == 0)]
    assert all(z.coverage_fraction == 0.0 for z in other_zones)


def test_compute_severity_report_overlapping_boxes_not_double_counted():
    # Two overlapping boxes covering the same 100x100 zone shouldn't push
    # coverage above 1.0 for that zone (union, not sum).
    boxes = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]])
    classes = np.array([0, 0])
    report = compute_severity_report(300, 300, boxes, classes, grid_rows=3, grid_cols=3)
    zone00 = next(z for z in report.zones if z.row == 0 and z.col == 0)
    assert zone00.coverage_fraction == 1.0


def test_rank_sites_orders_by_score_descending():
    low = compute_severity_report(300, 300, np.zeros((0, 4)), np.zeros((0,)))
    high = compute_severity_report(
        300, 300, np.array([[0.0, 0.0, 300.0, 300.0]]), np.array([0])
    )
    ranked = rank_sites([low, high])
    assert ranked[0] is high
    assert ranked[1] is low
