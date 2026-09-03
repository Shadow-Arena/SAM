"""Deterministic fake engine used when ``SAM_MOCK=true``.

No model is downloaded; used for UI development and tests. It mimics the
:class:`~sam3_studio.engine.sam3.SegmentationEngine` interface and returns
synthetic masks.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import ModeChoice, SamSettings
from ..domain import MaskInstance, PromptSet, SegmentationResult
from .common import ProgressCallback, merge_instances, validate_prompt


class MockSegmentationEngine:
    """Synthetic engine: same interface, no model weights."""

    device = "cpu"
    torch_dtype = None
    loaded = True
    last_load_error = None

    def __init__(self, settings: SamSettings | None = None):
        self.settings = settings or SamSettings()
        self.settings.apply_hf_environment()

    def ensure_pcs(self, progress: ProgressCallback = None) -> tuple:
        return (None, None)

    def ensure_tracker(self, progress: ProgressCallback = None) -> tuple:
        return (None, None)

    def segment(
        self,
        image: Image.Image,
        prompt: PromptSet,
        mode: ModeChoice | str = ModeChoice.AUTO,
        score_threshold: float | None = None,
        mask_threshold: float | None = None,
        max_masks: int | None = None,
        progress: ProgressCallback = None,
    ) -> SegmentationResult:
        image = image.convert("RGB")
        w, h = image.size
        mode = ModeChoice(mode)
        if mode == ModeChoice.AUTO:
            mode = (
                ModeChoice.TEXT
                if (prompt.has_text or prompt.has_boxes)
                else ModeChoice.POINT
                if prompt.has_points
                else ModeChoice.TEXT
            )
        validate_prompt(mode, prompt)
        rng = np.random.default_rng(abs(hash(prompt.describe() + str(mode))) % (2**32))
        count = 1 + (len(prompt.points_positive) if prompt.points_positive else 0) + int(bool(prompt.text))
        count = min(max(count, 2), 5)
        max_masks = self.settings.max_masks if max_masks is None else int(max_masks)
        instances: list[MaskInstance] = []
        for i in range(min(count, max_masks)):
            cx, cy = rng.integers(int(0.15 * w), int(0.85 * w)), rng.integers(int(0.15 * h), int(0.85 * h))
            rw, rh = int(rng.integers(int(0.08 * w), int(0.35 * w))), int(rng.integers(int(0.08 * h), int(0.35 * h)))
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx) / max(rw, 1)) ** 2 + ((yy - cy) / max(rh, 1)) ** 2 <= 1.0
            ys, xs = np.nonzero(mask)
            score = float(rng.random() * 0.4 + 0.5)
            source = "point" if prompt.points_positive and not prompt.text else "text" if prompt.text else "box"
            instances.append(
                MaskInstance(
                    mask=mask,
                    score=score,
                    box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    source=source,
                    label=f"mock {i + 1}",
                )
            )
        instances = merge_instances(instances, 0.9)[:max_masks]
        return SegmentationResult(
            instances=instances,
            image_size=(h, w),
            elapsed_seconds=0.01 * count,
            warnings=list(prompt.warnings) + ["MOCK ENGINE — results are synthetic."],
        )
