"""Shared helpers for SAM3 and mock engines: prompt validation and merging."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from ..config import ModeChoice
from ..domain import MaskInstance, PromptSet
from .errors import SegmentationError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None] | None

_SOURCE_ORDER = ["text", "box", "point"]


def report(progress: ProgressCallback, message: str) -> None:
    """Log a progress message and forward it to an optional callback."""
    logger.info(message)
    if progress is not None:
        try:
            progress(message)
        except Exception:  # noqa: BLE001 - progress callbacks must never break inference
            pass


def merge_sources(a: str, b: str) -> str:
    """Combine source labels in canonical order, e.g. 'text' + 'point' -> 'text+point'."""
    parts: list[str] = []
    for part in [a, b]:
        for token in part.split("+"):
            if token and token not in parts:
                parts.append(token)
    parts.sort(key=lambda token: _SOURCE_ORDER.index(token) if token in _SOURCE_ORDER else len(_SOURCE_ORDER))
    return "+".join(parts)


def validate_prompt(mode: ModeChoice, prompt: PromptSet) -> None:
    """Raise :class:`SegmentationError` for invalid prompt/mode combinations."""
    if mode == ModeChoice.TEXT and not prompt.has_text and not prompt.has_boxes:
        raise SegmentationError("Text mode needs a text prompt and/or box prompt.")
    if mode == ModeChoice.BOX and not prompt.has_boxes:
        raise SegmentationError("Box mode needs at least one box annotation.")
    if mode == ModeChoice.POINT and not prompt.points_positive:
        raise SegmentationError("Point mode needs at least one positive point annotation.")
    if mode == ModeChoice.MIXED and not (prompt.has_text or prompt.has_boxes or prompt.points_positive):
        raise SegmentationError("Mixed mode needs a text/box prompt and/or point prompt.")


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def merge_instances(instances: Sequence[MaskInstance], iou_threshold: float = 0.7) -> list[MaskInstance]:
    """Deduplicate overlapping instances, keeping the highest score."""
    ordered = sorted(
        enumerate(instances),
        key=lambda pair: pair[1].score if pair[1].score is not None else -1.0,
        reverse=True,
    )
    kept: list[MaskInstance] = []
    for _, inst in ordered:
        dup_slot: int | None = None
        best_iou = 0.0
        for slot, other in enumerate(kept):
            iou = _bbox_iou(inst.mask, other.mask)
            if iou > best_iou:
                best_iou, dup_slot = iou, slot
        if dup_slot is not None and best_iou > iou_threshold:
            # Merge: record the combined source on the kept (higher-score) mask.
            kept[dup_slot].source = merge_sources(kept[dup_slot].source, inst.source)
            if inst.score is not None and (kept[dup_slot].score is None or inst.score > kept[dup_slot].score):
                kept[dup_slot].score = inst.score
        else:
            kept.append(inst)
    for idx, inst in enumerate(kept):
        inst.object_id = idx + 1
    return kept
