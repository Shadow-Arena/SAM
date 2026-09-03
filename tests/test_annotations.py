from __future__ import annotations

from app.annotations import cluster_positive_points, negative_point_to_box


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
