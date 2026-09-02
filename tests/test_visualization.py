from __future__ import annotations

from app.schemas import MaskInstance, SegmentationResult
from app.visualization import gallery_item, overlay_masks, overlay_semantic, save_result


def test_overlay_masks(sample_image):
    import numpy as np

    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 100:220] = True
    inst = MaskInstance(mask=mask, score=0.87, box=(100, 50, 220, 150), source="text")
    overlay = overlay_masks(sample_image, [inst], opacity=0.5)
    assert overlay.size == sample_image.size
    assert overlay.mode == "RGB"
    arr = np.asarray(overlay)
    # overlay changed the masked region
    assert not np.array_equal(arr[50:150, 100:220], np.asarray(sample_image)[50:150, 100:220])


def test_overlay_semantic(sample_image):
    import numpy as np

    sem = np.zeros((200, 300), dtype=bool)
    sem[20:40, 20:40] = True
    out = overlay_semantic(sample_image, sem)
    assert out.size == sample_image.size


def test_gallery_item(sample_image):
    import numpy as np

    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 100:220] = True
    item = gallery_item(sample_image, MaskInstance(mask=mask, score=0.5, box=(100, 50, 220, 150), source="point"))
    assert item.size == sample_image.size


def test_save_result(sample_image, settings):
    import numpy as np

    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 100:220] = True
    result = SegmentationResult(
        instances=[MaskInstance(mask=mask, score=0.9, box=(100, 50, 220, 150), source="text")],
        image_size=(200, 300),
        elapsed_seconds=0.1,
        semantic_mask=mask,
    )
    paths = save_result(sample_image, result, settings.output_dir, "run_test")
    import os

    for key in ["composite", "json", "semantic"]:
        assert os.path.exists(paths[key])
    assert isinstance(paths["masks"], list) and len(paths["masks"]) == 1
