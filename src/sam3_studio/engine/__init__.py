"""Segmentation engines: real SAM3, mock, factory and shared helpers."""

from .common import ProgressCallback, merge_instances, merge_sources, validate_prompt
from .errors import SegmentationError
from .factory import Engine, create_engine, get_engine
from .mock import MockSegmentationEngine
from .sam3 import SegmentationEngine

__all__ = [
    "Engine",
    "ProgressCallback",
    "MockSegmentationEngine",
    "SegmentationEngine",
    "SegmentationError",
    "create_engine",
    "get_engine",
    "merge_instances",
    "merge_sources",
    "validate_prompt",
]
