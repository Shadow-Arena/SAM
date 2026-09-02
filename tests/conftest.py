from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.config import SamSettings


@pytest.fixture
def settings(tmp_path) -> SamSettings:
    return SamSettings(
        output_dir=tmp_path / "outputs",
        mock=True,
        lazy_load=False,
    )


@pytest.fixture
def sample_image() -> Image.Image:
    arr = np.zeros((200, 300, 3), dtype=np.uint8)
    arr[..., 0] = 40
    arr[..., 1] = 90
    arr[..., 2] = 160
    arr[40:120, 80:220] = (200, 220, 240)
    return Image.fromarray(arr, "RGB")
