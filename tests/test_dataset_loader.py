import numpy as np

from src.data.dataset_loader import (
    parse_yolo_label_file,
    xyxy_to_yolo_norm,
    yolo_norm_to_xyxy,
)


def test_yolo_norm_xyxy_round_trip():
    boxes_xyxy = np.array([[10.0, 20.0, 110.0, 220.0]])
    img_w, img_h = 640, 480

    norm = xyxy_to_yolo_norm(boxes_xyxy, img_w, img_h)
    recovered = yolo_norm_to_xyxy(norm, img_w, img_h)

    np.testing.assert_allclose(recovered, boxes_xyxy, atol=1e-3)


def test_yolo_norm_to_xyxy_empty_input():
    empty = np.zeros((0, 4), dtype=np.float32)
    result = yolo_norm_to_xyxy(empty, 640, 480)
    assert result.shape == (0, 4)


def test_parse_yolo_label_file_missing_returns_empty(tmp_path):
    boxes, classes = parse_yolo_label_file(tmp_path / "missing.txt")
    assert boxes.shape == (0, 4)
    assert classes.shape == (0,)


def test_parse_yolo_label_file_reads_lines(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n")

    boxes, classes = parse_yolo_label_file(label_path)
    assert boxes.shape == (2, 4)
    np.testing.assert_array_equal(classes, [0, 2])
    np.testing.assert_allclose(boxes[0], [0.5, 0.5, 0.2, 0.2])
