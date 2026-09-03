"""Engine factory + process-wide singleton."""

from __future__ import annotations

import threading
from typing import TypeAlias

from ..config import SamSettings
from .mock import MockSegmentationEngine
from .sam3 import SegmentationEngine

Engine: TypeAlias = SegmentationEngine | MockSegmentationEngine


def create_engine(settings: SamSettings | None = None) -> Engine:
    """Factory used across the app (CLI + API)."""
    settings = settings or SamSettings()
    if settings.mock:
        return MockSegmentationEngine(settings)
    return SegmentationEngine(settings)


_GLOBAL_ENGINE: Engine | None = None
_GLOBAL_LOCK = threading.Lock()


def get_engine(settings: SamSettings | None = None) -> Engine:
    """Process-wide engine singleton (models are heavy)."""
    global _GLOBAL_ENGINE
    with _GLOBAL_LOCK:
        if _GLOBAL_ENGINE is None:
            _GLOBAL_ENGINE = create_engine(settings or SamSettings())
        return _GLOBAL_ENGINE


__all__ = ["Engine", "create_engine", "get_engine"]
