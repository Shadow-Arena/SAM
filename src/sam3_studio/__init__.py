"""SAM3 Segment Studio — interactive promptable image segmentation.

Public package exports.
"""

from .config import SamSettings
from .engine import SegmentationEngine, create_engine, get_engine

__all__ = ["SamSettings", "SegmentationEngine", "create_engine", "get_engine", "__version__"]

__version__ = "0.2.0"
