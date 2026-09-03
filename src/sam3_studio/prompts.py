"""Prompt geometry helpers shared by the segmentation engine.

The web UI sends structured prompts directly (JSON points/boxes), so this
module only keeps the pure helpers used when building tracker/PCS inputs.
"""

from __future__ import annotations

from .domain import XYXY, Point


def cluster_positive_points(points: list[Point], max_distance: int) -> list[list[Point]]:
    """Greedy agglomerative clustering of points by Euclidean distance.

    Points closer than ``max_distance`` belong to the same object.
    """
    if not points:
        return []
    clusters: list[list[Point]] = [[points[0]]]
    centres: list[Point] = [points[0]]
    for point in points[1:]:
        best, best_dist = -1, float("inf")
        for i, centre in enumerate(centres):
            dist = ((point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist = i, dist
        if best_dist <= max_distance:
            clusters[best].append(point)
            cx = sum(p[0] for p in clusters[best]) / len(clusters[best])
            cy = sum(p[1] for p in clusters[best]) / len(clusters[best])
            centres[best] = (int(round(cx)), int(round(cy)))
        else:
            clusters.append([point])
            centres.append(point)
    return clusters


def negative_point_to_box(point: Point, image_size: tuple[int, int], relative: float) -> XYXY:
    """Expand a negative point into a small negative box (for SAM3 PCS prompts)."""
    h, w = image_size
    half_w = max(1, int(round(relative * w / 2)))
    half_h = max(1, int(round(relative * h / 2)))
    x0 = max(0, point[0] - half_w)
    x1 = min(w - 1, point[0] + half_w)
    y0 = max(0, point[1] - half_h)
    y1 = min(h - 1, point[1] + half_h)
    return (x0, y0, x1, y1)


def _cluster_to_points(cluster: list[Point]) -> list[list[float]]:
    """Normalize a cluster to ``[[x, y], ...]``.

    ``cluster_positive_points`` returns ``[object][point]``, but tolerate the
    flat ``[point]`` form as well so the tracker keeps working regardless of
    which representation arrives (defensive; the processors reject a wrong
    nesting depth with an opaque error otherwise).
    """
    if cluster and isinstance(cluster[0], (list, tuple)):
        return [[float(p[0]), float(p[1])] for p in cluster]
    return [[float(cluster[0]), float(cluster[1])]]


def build_tracker_prompts(
    points_positive: list[Point],
    points_negative: list[Point],
    cluster_distance_px: int,
) -> tuple[list, list]:
    """Build SAM3 tracker processor inputs.

    Returns ``(input_points, input_labels)`` with the nesting the
    ``Sam3TrackerProcessor`` requires (transformers 5.x):

    * ``input_points`` = ``[image][object][point][x, y]``  (4 levels)
    * ``input_labels`` = ``[image][object][label]``        (3 levels)

    Negative points are attached to the nearest positive cluster.
    """
    clusters = cluster_positive_points(list(points_positive), cluster_distance_px)
    objects: list[list[list[float]]] = [_cluster_to_points(c) for c in clusters]
    if not objects:
        return [], []
    centres = [
        (sum(p[0] for p in obj) / len(obj), sum(p[1] for p in obj) / len(obj)) for obj in objects
    ]
    labels: list[list[int]] = [[1] * len(obj) for obj in objects]
    for neg in points_negative:
        idx = min(
            range(len(objects)),
            key=lambda i: (neg[0] - centres[i][0]) ** 2 + (neg[1] - centres[i][1]) ** 2,
        )
        objects[idx].append([float(neg[0]), float(neg[1])])
        labels[idx].append(0)
    return [objects], [labels]


def build_pcs_prompts(
    boxes_positive: list[XYXY],
    boxes_negative: list[XYXY],
) -> tuple[list | None, list | None]:
    """Build SAM3 PCS processor inputs.

    Returns ``(input_boxes, input_boxes_labels)`` with the nesting the
    ``Sam3Processor`` requires (transformers 5.x):

    * ``input_boxes``       = ``[image][box][x1, y1, x2, y2]`` (3 levels)
    * ``input_boxes_labels`` = ``[image][box]``                (2 levels)

    Labels are ``1`` for positive boxes and ``0`` for negative boxes.
    Returns ``(None, None)`` when there are no boxes.
    """
    boxes = [[float(v) for v in b] for b in list(boxes_positive) + list(boxes_negative)]
    if not boxes:
        return None, None
    labels = [1] * len(boxes_positive) + [0] * len(boxes_negative)
    return [boxes], [labels]


__all__ = [
    "build_pcs_prompts",
    "build_tracker_prompts",
    "cluster_positive_points",
    "negative_point_to_box",
]
