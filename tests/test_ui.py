from __future__ import annotations

from PIL import Image, ImageDraw

from app.annotations import NEGATIVE_COLOR, POSITIVE_COLOR
from app.segmentation import MockSegmentationEngine
from app.ui import MODE_INFO, build_app, make_segment_handler


def test_build_app(settings):
    demo = build_app(settings, engine_factory=MockSegmentationEngine)
    assert demo is not None
    assert len(demo.blocks) > 3


def test_segment_handler_text_mode(settings, sample_image):
    handler = make_segment_handler(settings, engine_factory=MockSegmentationEngine)
    value = {"background": sample_image, "layers": [], "composite": sample_image}
    composite, gallery, rows, files, status = handler(value, "Auto", "car", 0.3, 0.5, 0.5, 10, False)
    assert composite.size == sample_image.size
    assert len(gallery) == 2
    assert len(rows) == 2
    assert "✅" in status
    assert len(files) > 0


def test_segment_handler_point_draw(settings, sample_image):
    layer = Image.new("RGBA", sample_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse([95, 95, 105, 105], fill=POSITIVE_COLOR + (255,))
    draw.ellipse([25, 25, 35, 35], fill=NEGATIVE_COLOR + (255,))
    value = {"background": sample_image, "layers": [layer], "composite": sample_image}
    handler = make_segment_handler(settings, engine_factory=MockSegmentationEngine)
    composite, gallery, rows, files, status = handler(value, "Point", "", 0.3, 0.5, 0.5, 10, False)
    assert composite is not None
    assert rows and "❌" not in status


def test_segment_handler_no_input(settings, sample_image):
    handler = make_segment_handler(settings, engine_factory=MockSegmentationEngine)
    value = {"background": sample_image, "layers": [], "composite": sample_image}
    composite, gallery, rows, files, status = handler(value, "Auto", "", 0.3, 0.5, 0.5, 10, False)
    assert composite is None
    assert "⚠️" in status


def test_segment_handler_no_image(settings):
    handler = make_segment_handler(settings, engine_factory=MockSegmentationEngine)
    composite, gallery, rows, files, status = handler(None, "Auto", "cat", 0.3, 0.5, 0.5, 10, False)
    assert composite is None and "⚠️" in status


def test_mode_info_keys():
    assert set(MODE_INFO) == {"Auto", "Text", "Box", "Point", "Mixed"}
