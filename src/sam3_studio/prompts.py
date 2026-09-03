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


__all__ = ["cluster_positive_points", "negative_point_to_box"]
