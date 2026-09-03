"""API route modules."""

from .segment import router as segment_router
from .system import router as system_router

__all__ = ["segment_router", "system_router"]
