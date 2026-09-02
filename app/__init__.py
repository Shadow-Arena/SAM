"""SAM3 Segment Studio — interactive promptable image segmentation.

Public package exports.
"""

from .config import SamSettings
from .segmentation import SegmentationEngine, create_engine

__all__ = ["SamSettings", "SegmentationEngine", "create_engine", "__version__"]

__version__ = "0.1.0"
