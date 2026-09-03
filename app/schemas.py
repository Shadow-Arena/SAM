"""Shared data structures for prompts and segmentation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

XYXY = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass
class PromptAnnotation:
    """One stroke extracted from the image editor."""

    kind: Literal["point", "box"]
    xyxy: XYXY
    point: Point | None = None
    positive: bool = True
    color: tuple[int, int, int] = (0, 0, 0)
    area: int = 0


@dataclass
class PromptSet:
    """Normalized set of prompts derived from UI/CLI input."""

    text: str | None = None
    points_positive: list[Point] = field(default_factory=list)
    points_negative: list[Point] = field(default_factory=list)
    boxes_positive: list[XYXY] = field(default_factory=list)
    boxes_negative: list[XYXY] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(
            self.text or self.points_positive or self.points_negative or self.boxes_positive or self.boxes_negative
        )

    @property
    def has_text(self) -> bool:
        return bool(self.text)

    @property
    def has_boxes(self) -> bool:
        return bool(self.boxes_positive or self.boxes_negative)

    @property
    def has_points(self) -> bool:
        return bool(self.points_positive or self.points_negative)

    def describe(self) -> str:
        parts = []
        if self.text:
            parts.append(f'text="{self.text}"')
        if self.boxes_positive:
            parts.append(f"{len(self.boxes_positive)}+ box(es)")
        if self.boxes_negative:
            parts.append(f"{len(self.boxes_negative)}- box(es)")
        if self.points_positive:
            parts.append(f"{len(self.points_positive)}+ point(s)")
        if self.points_negative:
            parts.append(f"{len(self.points_negative)}- point(s)")
        return ", ".join(parts) or "no prompts"


@dataclass
class MaskInstance:
    """One segmented object instance."""

    mask: np.ndarray  # bool [H, W] at original image size
    score: float | None
    box: XYXY
    source: str  # e.g. "text", "box", "point", "text+box", "text+point"
    label: str = ""
    object_id: int = 0

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    def to_record(self) -> dict:
        return {
            "id": self.object_id,
            "label": self.label,
            "source": self.source,
            "score": self.score,
            "box": list(self.box),
            "area_px": self.area_px,
        }


@dataclass
class SegmentationResult:
    """Full result of one segmentation request."""

    instances: list[MaskInstance]
    image_size: tuple[int, int] = (0, 0)
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    semantic_mask: np.ndarray | None = None  # bool or float mask for "text" mode

    def to_json(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "num_instances": len(self.instances),
            "warnings": self.warnings,
            "instances": [i.to_record() for i in self.instances],
            "has_semantic_mask": self.semantic_mask is not None,
        }
