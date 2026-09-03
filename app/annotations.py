"""Extract points/boxes/colors from image-editor annotation strokes.

Gradio's ``ImageEditor`` returns the background plus a list of "layers".
Every brush stroke / box / point drawn by the user becomes one or more opaque
connected components inside those layers. We convert them into structured
``PromptAnnotation`` objects:

* small strokes  -> points  (click centroid),
* large strokes  -> boxes   (bounding box of the stroke),
* green strokes  -> positive prompts,
* red strokes    -> negative prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import SamSettings
from .schemas import XYXY, Point, PromptAnnotation, PromptSet

# Fixed color palette used by the UI brush (green = positive, red = negative).
POSITIVE_COLOR = (50, 204, 112)
NEGATIVE_COLOR = (204, 50, 50)

COLOR_TOLERANCE = 18


@dataclass
class Component:
    """A connected annotation stroke."""

    xyxy: XYXY
    centroid: Point
    area: int
    rgb: tuple[int, int, int]


def _find(mapping: list[int], x: int) -> int:
    """Union-find find with path compression."""
    root = x
    while mapping[root] != root:
        root = mapping[root]
    while mapping[x] != root:
        mapping[x], x = root, mapping[x]
    return root


def _union(mapping: list[int], a: int, b: int) -> None:
    ra, rb = _find(mapping, a), _find(mapping, b)
    if ra != rb:
        mapping[rb] = ra


def connected_components(mask: np.ndarray) -> list[Component]:
    """Label 4-connected components with a two-pass union-find.

    ``mask`` is a boolean 2D array. Returns components containing their bbox,
    centroid and area, in reading order.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return []
    labels = np.full((h, w), -1, dtype=np.int64)
    parent: list[int] = []
    next_label = 0

    def new_label() -> int:
        nonlocal next_label
        parent.append(next_label)
        next_label += 1
        return next_label - 1

    # First pass: provisional labels + unions for right/down neighbors.
    for y in range(h):
        row = mask[y]
        up = labels[y - 1] if y else None
        for x in range(w):
            if not row[x]:
                continue
            left = labels[y, x - 1] if x else -1
            upv = up[x] if up is not None else -1
            if left < 0 and upv < 0:
                labels[y, x] = new_label()
            elif left >= 0 and upv >= 0 and left != upv:
                labels[y, x] = left
                _union(parent, int(left), int(upv))
            elif left >= 0:
                labels[y, x] = left
            else:
                labels[y, x] = upv

    # Second pass: resolve all labels (background stays at -1).
    total = next_label
    if total == 0:
        return []
    flat = np.array([_find(parent, i) for i in range(total)], dtype=np.int64)
    resolved = np.where(labels >= 0, flat[np.clip(labels, 0, None)], -1)
    # Normalize ids to 0..n-1 by first-seen order (back onto the grid).
    foreground = resolved >= 0
    _, inverse = np.unique(resolved[foreground], return_inverse=True)
    order = np.full(resolved.shape, -1, dtype=np.int64)
    order[foreground] = inverse
    components: list[Component] = []
    for comp_id in range(order.max() + 1):
        ys, xs = np.nonzero(order == comp_id)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))
        components.append(
            Component(
                xyxy=(x0, y0, x1, y1),
                centroid=(cx, cy),
                area=int(ys.size),
                rgb=(0, 0, 0),
            )
        )
    return components


def layer_component_rgb(layer: Image.Image, component: Component) -> tuple[int, int, int]:
    """Average RGB of the opaque pixels of a component inside an RGBA layer."""
    rgba = np.asarray(layer.convert("RGBA"), dtype=np.int32)
    x0, y0, x1, y1 = component.xyxy
    patch = rgba[y0 : y1 + 1, x0 : x1 + 1]
    alpha = patch[..., 3] > 0
    if not alpha.any():
        return (0, 0, 0)
    rgb = patch[..., :3][alpha]
    return tuple(int(round(float(c))) for c in rgb.mean(axis=0))


def classify_sign(rgb: tuple[int, int, int], settings: SamSettings) -> bool:
    """Return True for a positive (green) stroke, False for negative (red)."""
    r, g, b = rgb
    if g - max(r, b) >= COLOR_TOLERANCE:
        return True
    if r - max(g, b) >= COLOR_TOLERANCE:
        return False
    return settings.unknown_color_label == "positive"


def _is_point(component: Component, image_size: tuple[int, int], settings: SamSettings) -> bool:
    h, w = image_size
    x0, y0, x1, y1 = component.xyxy
    width, height = x1 - x0 + 1, y1 - y0 + 1
    max_dim = max(width, height)
    abs_limit = settings.point_max_size_px
    rel_limit = int(settings.point_max_size_relative * min(h, w))
    return max_dim <= max(abs_limit, rel_limit)


def analyze_layers(
    layers: Sequence[Image.Image | np.ndarray],
    image_size: tuple[int, int],
    settings: SamSettings,
) -> tuple[list[PromptAnnotation], list[str]]:
    """Convert editor layers into prompt annotations + warnings."""
    annotations: list[PromptAnnotation] = []
    warnings: list[str] = []
    for idx, layer in enumerate(layers):
        if layer is None:
            continue
        img = layer if isinstance(layer, Image.Image) else Image.fromarray(np.asarray(layer))
        rgba = np.asarray(img.convert("RGBA"))
        alpha = rgba[..., 3] > 0
        if not alpha.any():
            continue
        for comp in connected_components(alpha):
            if comp.area < settings.min_component_area:
                continue
            comp.rgb = layer_component_rgb(img, comp)
            positive = classify_sign(comp.rgb, settings)
            if _is_point(comp, image_size, settings):
                point = (
                    min(max(comp.centroid[0], 0), image_size[1] - 1),
                    min(max(comp.centroid[1], 0), image_size[0] - 1),
                )
                annotations.append(
                    PromptAnnotation(
                        kind="point", xyxy=comp.xyxy, point=point, positive=positive, color=comp.rgb, area=comp.area
                    )
                )
            else:
                annotations.append(
                    PromptAnnotation(kind="box", xyxy=comp.xyxy, positive=positive, color=comp.rgb, area=comp.area)
                )
                if comp.area / max(1, (comp.xyxy[2] - comp.xyxy[0] + 1) * (comp.xyxy[3] - comp.xyxy[1] + 1)) > 0.55:
                    warnings.append(
                        f"Layer {idx + 1}: a large filled stroke was used; its bounding box is used as a box prompt."
                    )
    if not annotations:
        warnings.append("No annotation strokes detected.")
    return annotations, warnings


def annotations_to_prompt_set(annotations: Sequence[PromptAnnotation], text: str | None = None) -> PromptSet:
    """Group annotations into a :class:`PromptSet`."""
    prompt = PromptSet(text=(text or "").strip() or None)
    for ann in annotations:
        if ann.kind == "point":
            assert ann.point is not None
            (prompt.points_positive if ann.positive else prompt.points_negative).append(ann.point)
        else:
            (prompt.boxes_positive if ann.positive else prompt.boxes_negative).append(ann.xyxy)
    if prompt.points_negative and not prompt.points_positive:
        prompt.warnings.append("Negative points without a positive point are ignored unless a text prompt is given.")
    return prompt


def normalize_editor_value(value) -> tuple[Image.Image | None, list[Image.Image]]:
    """Accept the editor value from Gradio 5 (tuple) or Gradio 6 (dict).

    Returns ``(background, layers)`` as PIL RGBA images.
    """
    background = None
    layers: list[Image.Image] = []
    if value is None:
        return background, layers
    if isinstance(value, dict):
        bg = value.get("background")
        layers = value.get("layers") or []
        if bg is not None:
            background = bg if isinstance(bg, Image.Image) else Image.fromarray(np.asarray(bg)).convert("RGB")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        bg = value[0]
        layers = value[1] or []
        if bg is not None:
            background = bg if isinstance(bg, Image.Image) else Image.fromarray(np.asarray(bg)).convert("RGB")
    else:
        background = value if isinstance(value, Image.Image) else Image.fromarray(np.asarray(value)).convert("RGB")
        layers = []
    norm_layers: list[Image.Image] = []
    for layer in layers:
        if layer is None:
            continue
        norm_layers.append(
            layer if isinstance(layer, Image.Image) else Image.fromarray(np.asarray(layer)).convert("RGBA")
        )
    if background is not None:
        background = background.convert("RGB")
    return background, norm_layers


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
