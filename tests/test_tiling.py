import numpy as np

from src.data.tiling import (
    TileCoord,
    clip_boxes_to_tile,
    compute_tile_coords,
    merge_tile_detections,
    reproject_boxes_to_full_image,
)


def test_compute_tile_coords_covers_full_image_with_no_gaps():
    coords = compute_tile_coords(img_w=1000, img_h=800, tile_size=400, overlap=0.2)
    for c in coords:
        assert c.x_offset + c.width <= 1000
        assert c.y_offset + c.height <= 800

    max_x = max(c.x_offset + c.width for c in coords)
    max_y = max(c.y_offset + c.height for c in coords)
    assert max_x == 1000
    assert max_y == 800


def test_compute_tile_coords_small_image_returns_single_tile():
    coords = compute_tile_coords(img_w=300, img_h=200, tile_size=640, overlap=0.2)
    assert len(coords) == 1
    assert coords[0].width == 300
    assert coords[0].height == 200


def test_clip_boxes_to_tile_keeps_fully_inside_box():
    tile = TileCoord(tile_index=0, x_offset=100, y_offset=100, width=200, height=200)
    boxes = np.array([[150.0, 150.0, 180.0, 180.0]])
    classes = np.array([2])
    kept_boxes, kept_classes = clip_boxes_to_tile(boxes, classes, tile, min_visibility=0.3)
    assert kept_boxes.shape == (1, 4)
    np.testing.assert_allclose(kept_boxes[0], [50, 50, 80, 80])
    assert kept_classes[0] == 2


def test_clip_boxes_to_tile_drops_mostly_outside_box():
    tile = TileCoord(tile_index=0, x_offset=0, y_offset=0, width=100, height=100)
    boxes = np.array([[90.0, 90.0, 300.0, 300.0]])  # only a sliver overlaps
    classes = np.array([0])
    kept_boxes, kept_classes = clip_boxes_to_tile(boxes, classes, tile, min_visibility=0.3)
    assert kept_boxes.shape == (0, 4)


def test_reproject_boxes_to_full_image():
    tile = TileCoord(tile_index=0, x_offset=50, y_offset=100, width=200, height=200)
    boxes = np.array([[10.0, 10.0, 40.0, 40.0]])
    reprojected = reproject_boxes_to_full_image(boxes, tile)
    np.testing.assert_allclose(reprojected[0], [60, 110, 90, 140])


def test_merge_tile_detections_dedupes_overlapping_boxes_across_tiles():
    tile_a = TileCoord(tile_index=0, x_offset=0, y_offset=0, width=200, height=200)
    tile_b = TileCoord(tile_index=1, x_offset=100, y_offset=0, width=200, height=200)

    # Same real-world object seen near the boundary of both tiles.
    boxes_a = [np.array([[80.0, 80.0, 120.0, 120.0]])]
    boxes_b = [np.array([[-10.0, 80.0, 30.0, 120.0]])]  # tile-local coords in tile_b
    scores = [np.array([0.9]), np.array([0.8])]
    classes = [np.array([0]), np.array([0])]

    merged = merge_tile_detections(
        [boxes_a[0], boxes_b[0]], scores, classes, [tile_a, tile_b], iou_threshold=0.5
    )
    assert merged["boxes"].shape[0] == 1
    assert merged["scores"][0] == 0.9


def test_merge_tile_detections_keeps_distinct_boxes():
    tile_a = TileCoord(tile_index=0, x_offset=0, y_offset=0, width=200, height=200)
    boxes = [np.array([[10.0, 10.0, 30.0, 30.0], [150.0, 150.0, 180.0, 180.0]])]
    scores = [np.array([0.9, 0.85])]
    classes = [np.array([0, 1])]

    merged = merge_tile_detections(boxes, scores, classes, [tile_a], iou_threshold=0.5)
    assert merged["boxes"].shape[0] == 2
