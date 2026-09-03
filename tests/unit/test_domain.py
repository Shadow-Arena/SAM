from __future__ import annotations

import numpy as np

from sam3_studio.domain import MaskInstance, PromptSet, SegmentationResult


def test_prompt_set_bool():
    assert not PromptSet()
    assert PromptSet(text="car")
    assert PromptSet(points_positive=[(1, 2)])
    assert PromptSet(boxes_negative=[(1, 2, 3, 4)])


def test_prompt_set_properties():
    p = PromptSet(
        text="bus",
        points_positive=[(10, 10)],
        points_negative=[(20, 20)],
        boxes_positive=[(1, 2, 3, 4)],
        boxes_negative=[(5, 6, 7, 8)],
    )
    assert p.has_text and p.has_points and p.has_boxes


def test_prompt_set_describe():
    p = PromptSet(text="car", points_positive=[(1, 1)], boxes_negative=[(1, 2, 3, 4)])
    desc = p.describe()
    assert 'text="car"' in desc
    assert "1+ point(s)" in desc
    assert "1- box(es)" in desc


def test_mask_instance_record():
    mask = np.zeros((50, 60), dtype=bool)
    mask[10:30, 20:40] = True
    inst = MaskInstance(mask=mask, score=0.75, box=(20, 10, 40, 30), source="text", object_id=3)
    record = inst.to_record()
    assert record == {
        "id": 3,
        "label": "",
        "source": "text",
        "score": 0.75,
        "box": [20, 10, 40, 30],
        "area_px": 400,
    }


def test_result_json_shape():
    result = SegmentationResult(
        instances=[],
        image_size=(200, 300),
        elapsed_seconds=0.5,
        warnings=["w"],
        semantic_mask=None,
    )
    payload = result.to_json()
    assert set(payload) == {
        "image_size",
        "elapsed_seconds",
        "num_instances",
        "warnings",
        "instances",
        "has_semantic_mask",
    }
    assert payload["image_size"] == [200, 300]
