from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sam3_studio.prompts import (
    build_pcs_prompts,
    build_tracker_prompts,
    cluster_positive_points,
    negative_point_to_box,
)


def test_cluster_positive_points():
    clusters = cluster_positive_points([(10, 10), (14, 14), (200, 200)], 48)
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_cluster_single_point():
    assert cluster_positive_points([(5, 5)], 10) == [[(5, 5)]]
    assert cluster_positive_points([], 10) == []


def test_negative_point_to_box():
    box = negative_point_to_box((100, 80), (200, 300), 0.04)
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0 and x1 < 300 and y1 < 200
    assert x0 <= 100 <= x1 and y0 <= 80 <= y1


def test_negative_point_to_box_clamps_at_edges():
    box = negative_point_to_box((0, 0), (100, 100), 0.1)
    assert box[0] == 0 and box[1] == 0


# --------------------------------------------------------------------------
# Regression: the `transformers` 5.x SAM3 processors validate the nesting
# depth of prompt lists and reject wrong shapes with an opaque error
# (e.g. "Input points must be a nested list with 4 levels ... Got 3 levels").
# These tests lock the shapes our engine builds to the real processor
# contract WITHOUT downloading multi-GB weights.
# --------------------------------------------------------------------------

EXPECTED_TRACKER_POINTS = "[image level, object level, point level, point coordinates]"
EXPECTED_TRACKER_LABELS = "[image level, object level, point level]"
EXPECTED_PCS_BOXES = "[image level, box level, box coordinates]"
EXPECTED_PCS_LABELS = "[image level, box level]"


def _nesting_depth(value):
    depth = 0
    node = value
    while isinstance(node, list) and node:
        depth += 1
        node = node[0]
    return depth


def test_tracker_prompts_have_4_and_3_levels():
    points, labels = build_tracker_prompts([(404, 303)], [], 100)
    assert _nesting_depth(points) == 4
    assert _nesting_depth(labels) == 3
    # one image → one object → one point
    assert points == [[[[404.0, 303.0]]]]
    assert labels == [[[1]]]


def test_tracker_prompts_multi_object_and_negative():
    points, labels = build_tracker_prompts([(10, 10), (20, 20), (300, 300)], [(15, 15)], 48)
    assert _nesting_depth(points) == 4
    assert _nesting_depth(labels) == 3
    assert len(points[0]) == 2  # two positive clusters
    flat = [p for obj in points[0] for p in obj]
    assert len(flat) == 4  # 3 positive + 1 negative attached
    labels_flat = [v for obj in labels[0] for v in obj]
    assert sorted(labels_flat) == [0, 1, 1, 1]


def test_tracker_prompts_normalize_flat_clusters(monkeypatch):
    """If clustering returns flat [x, y] clusters (the shape that produced
    the user's 500: 'Got 3 levels'), the builder must still emit the
    4-level contract the processor requires."""

    def flat_clusters(points, max_distance):
        return [list(p) for p in points]  # [[x, y]] instead of [[[x, y]]]

    monkeypatch.setattr("sam3_studio.prompts.cluster_positive_points", flat_clusters)
    points, labels = build_tracker_prompts([(404, 303)], [], 100)
    assert _nesting_depth(points) == 4
    assert _nesting_depth(labels) == 3
    assert points == [[[[404.0, 303.0]]]]


def test_pcs_prompts_have_3_and_2_levels():
    boxes, labels = build_pcs_prompts([(10, 20, 300, 400)], [(5, 5, 8, 8)])
    assert _nesting_depth(boxes) == 3
    assert _nesting_depth(labels) == 2
    assert boxes == [[[10.0, 20.0, 300.0, 400.0], [5.0, 5.0, 8.0, 8.0]]]
    assert labels == [[1, 0]]


def test_pcs_prompts_empty():
    assert build_pcs_prompts([], []) == (None, None)


def _tracker_processor():
    from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
    from transformers.models.sam3_tracker.processing_sam3_tracker import Sam3TrackerProcessor

    return Sam3TrackerProcessor(image_processor=Sam3ImageProcessor(target_size=1024))


@pytest.mark.parametrize(
    "positive,negative",
    [
        ([(404, 303)], []),
        ([(80, 60), (200, 150)], [(50, 50)]),
    ],
)
def test_tracker_processor_accepts_engine_shapes(positive, negative):
    """Full real-processor pass (no weights): the exact lists the engine
    builds must be accepted by the sam3_tracker processor."""
    proc = _tracker_processor()
    points, labels = build_tracker_prompts(positive, negative, 100)
    img = Image.fromarray(np.zeros((600, 800, 3), dtype=np.uint8))
    out = proc(images=img, input_points=points, input_labels=labels, return_tensors="pt")
    obj_count = len(points[0])
    padded = max(len(obj) for obj in points[0])  # processor pads per-object
    assert out["input_points"].shape == (1, obj_count, padded, 2)
    assert out["input_labels"].shape == (1, obj_count, padded)


def test_tracker_processor_rejects_wrong_nesting():
    """Lock the processor contract: the old 3-level shape must fail exactly
    like the user's traceback (so this regression can never come back)."""
    from transformers.models.sam3_tracker.processing_sam3_tracker import Sam3TrackerProcessor

    proc = Sam3TrackerProcessor.__new__(Sam3TrackerProcessor)
    with pytest.raises(ValueError, match="4 levels"):
        proc._validate_single_input(
            [[[404, 303]]],
            expected_depth=4,
            input_name="points",
            expected_format=EXPECTED_TRACKER_POINTS,
            expected_coord_size=2,
        )


def test_pcs_processor_rejects_wrong_nesting():
    """Same contract lock for SAM3 PCS boxes/labels."""
    from transformers.models.sam3.processing_sam3 import Sam3Processor

    proc = Sam3Processor.__new__(Sam3Processor)
    with pytest.raises(ValueError, match="3 levels"):
        proc._validate_single_input(
            [[10, 20, 300, 400]],
            expected_depth=3,
            input_name="boxes",
            expected_format=EXPECTED_PCS_BOXES,
            expected_coord_size=4,
        )
    with pytest.raises(ValueError, match="2 levels"):
        proc._validate_single_input(
            [1, 0],
            expected_depth=2,
            input_name="labels",
            expected_format=EXPECTED_PCS_LABELS,
        )
