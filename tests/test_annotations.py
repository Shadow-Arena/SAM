from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.annotations import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    analyze_layers,
    annotations_to_prompt_set,
    classify_sign,
    cluster_positive_points,
    connected_components,
    negative_point_to_box,
    normalize_editor_value,
)
from app.config import SamSettings


def _layer_with_strokes(spots: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]]) -> Image.Image:
    """Draw strokes onto an RGBA layer: small dots, large rectangle outlines."""
    layer = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for (cx, cy), (w, h), color in spots:
        if w > 40:
            draw.rectangle([cx - w, cy - h, cx + w, cy + h], outline=color + (255,), width=4)
        else:
            draw.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=color + (255,))
    return layer


def test_connected_components():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[1, 1] = True
    mask[3, 3] = True
    comps = connected_components(mask)
    assert len(comps) == 2
    areas = sorted(c.area for c in comps)
    assert areas == [1, 3]


def test_analyze_layers_point_and_box(settings):
    layer = _layer_with_strokes(
        [
            ((50, 50), (10, 10), POSITIVE_COLOR),  # small -> positive point
            ((200, 100), (16, 16), NEGATIVE_COLOR),  # small -> negative point
            ((150, 120), (90, 60), POSITIVE_COLOR),  # large outline -> positive box
        ]
    )
    annotations, warnings = analyze_layers([layer], (200, 300), settings)
    kinds = sorted((a.kind, a.positive) for a in annotations)
    assert ("point", True) in kinds
    assert ("point", False) in kinds
    assert ("box", True) in kinds
    assert len(warnings) == 0


def test_prompt_set_from_annotations(settings):
    layer = _layer_with_strokes([((40, 40), (10, 10), POSITIVE_COLOR), ((260, 150), (36, 30), NEGATIVE_COLOR)])
    annotations, _ = analyze_layers([layer], (200, 300), settings)
    prompt = annotations_to_prompt_set(annotations, settings, text="cat")
    assert prompt.text == "cat"
    assert len(prompt.points_positive) == 1
    assert len(prompt.boxes_negative) == 1
    assert isinstance(prompt.boxes_negative[0], tuple)


def test_classify_sign():
    settings = SamSettings(_env_file=None)
    assert classify_sign(POSITIVE_COLOR, settings)
    assert not classify_sign(NEGATIVE_COLOR, settings)
    assert classify_sign((128, 128, 128), settings)  # unknown -> default positive


def test_cluster_positive_points():
    clusters = cluster_positive_points([(10, 10), (14, 14), (200, 200)], 48)
    assert len(clusters) == 2
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_negative_point_to_box():
    box = negative_point_to_box((100, 80), (200, 300), 0.04)
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0 and x1 < 300 and y1 < 200
    assert x0 <= 100 <= x1 and y0 <= 80 <= y1


def test_normalize_editor_value_dict_v6(sample_image):
    layer = Image.new("RGBA", sample_image.size, (0, 0, 0, 0))
    value = {"background": sample_image, "layers": [layer], "composite": sample_image}
    bg, layers = normalize_editor_value(value)
    assert bg is not None and bg.mode == "RGB"
    assert len(layers) == 1


def test_normalize_editor_value_tuple_v5(sample_image):
    value = (sample_image, [], sample_image)
    bg, layers = normalize_editor_value(value)
    assert bg is not None and layers == []
