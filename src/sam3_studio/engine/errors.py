"""Engine-level exceptions."""

from __future__ import annotations


class SegmentationError(RuntimeError):
    """Raised when a segmentation request cannot be executed."""
