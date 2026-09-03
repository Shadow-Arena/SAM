from __future__ import annotations

import numpy as np
import pytest

from sam3_studio.config import ModeChoice, SamSettings
from sam3_studio.domain import MaskInstance, PromptSet
from sam3_studio.engine import (
    MockSegmentationEngine,
    SegmentationError,
    create_engine,
    merge_instances,
    merge_sources,
)


def _mask(shape, x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_merge_sources():
    assert merge_sources("text", "point") == "text+point"
    assert merge_sources("text+box", "box") == "text+box"
    assert merge_sources("point", "text") == "text+point"


def test_merge_instances_dedupes():
    shape = (100, 100)
    a = MaskInstance(mask=_mask(shape, 10, 10, 60, 60), score=0.9, box=(10, 10, 60, 60), source="text")
    b = MaskInstance(mask=_mask(shape, 12, 12, 62, 62), score=0.5, box=(12, 12, 62, 62), source="point")
    c = MaskInstance(mask=_mask(shape, 70, 70, 90, 90), score=0.7, box=(70, 70, 90, 90), source="box")
    merged = merge_instances([a, b, c], iou_threshold=0.7)
    assert len(merged) == 2
    assert "text+point" in {i.source for i in merged}
    ids = [i.object_id for i in merged]
    assert ids == [1, 2]


def test_merge_instances_keeps_distinct():
    shape = (100, 100)
    a = MaskInstance(mask=_mask(shape, 10, 10, 30, 30), score=0.9, box=(10, 10, 30, 30), source="text")
    b = MaskInstance(mask=_mask(shape, 60, 60, 90, 90), score=0.8, box=(60, 60, 90, 90), source="point")
    merged = merge_instances([a, b], 0.7)
    assert len(merged) == 2


def test_mock_engine_text(sample_image, settings):
    engine = MockSegmentationEngine(settings)
    result = engine.segment(sample_image, PromptSet(text="car"), ModeChoice.TEXT)
    assert len(result.instances) == 2
    for inst in result.instances:
        assert inst.mask.shape == (200, 300)
        assert inst.box[0] <= inst.box[2] and inst.box[1] <= inst.box[3]


def test_mock_engine_points(sample_image, settings):
    engine = MockSegmentationEngine(settings)
    prompt = PromptSet(points_positive=[(100, 100)], points_negative=[(50, 50)])
    result = engine.segment(sample_image, prompt, ModeChoice.POINT)
    assert len(result.instances) >= 1
    assert any("mock" in i.source or i.source == "point" for i in result.instances)


def test_mixed_requires_prompt(sample_image, settings):
    engine = MockSegmentationEngine(settings)
    with pytest.raises(SegmentationError) as exc:
        engine.segment(sample_image, PromptSet(), ModeChoice.MIXED)
    assert "Mixed mode needs" in str(exc.value)


def test_result_json(sample_image, settings):
    engine = MockSegmentationEngine(settings)
    result = engine.segment(sample_image, PromptSet(text="bus"), ModeChoice.TEXT)
    payload = result.to_json()
    assert payload["num_instances"] == len(result.instances)


def test_max_masks_cap(sample_image, settings):
    engine = MockSegmentationEngine(settings)
    result = engine.segment(sample_image, PromptSet(text="x"), ModeChoice.TEXT, max_masks=1)
    assert len(result.instances) <= 1


def test_settings_env_mock_flag(settings):
    assert settings.mock is True
    assert isinstance(create_engine(settings), MockSegmentationEngine)
    real_settings = SamSettings(_env_file=None)
    # real engine would attempt downloads; just check factory type contract.
    assert real_settings.mock is False
