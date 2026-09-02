"""Application configuration built with pydantic-settings.

Every value can be overridden through the environment (``SAM_`` prefix) or the
``.env`` file, e.g. ``SAM_MODEL_ID=facebook/sam3``.

Run ``make config`` to print the effective configuration.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceChoice(str, Enum):
    """Inference device selection."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"


class DTypeChoice(str, Enum):
    """Model weight dtype selection."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ModeChoice(str, Enum):
    """Supported prompt modes."""

    AUTO = "auto"
    TEXT = "text"
    BOX = "box"
    POINT = "point"
    MIXED = "mixed"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class SamSettings(BaseSettings):
    """Central application settings (pydantic).

    Sources, in increasing priority:
    1. defaults defined here,
    2. ``.env`` file,
    3. process environment variables with the ``SAM_`` prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SAM_",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ model
    model_id: str = Field(default="facebook/sam3", description="SAM3 PCS model on the HF Hub or a local path.")
    tracker_model_id: str | None = Field(
        default=None, description="SAM3 tracker (PVS) model. Falls back to `model_id`."
    )
    device: DeviceChoice = DeviceChoice.AUTO
    dtype: DTypeChoice = DTypeChoice.AUTO
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    local_files_only: bool = Field(default=False, description="Do not contact the hub; load from local cache only.")
    lazy_load: bool = Field(default=True, description="Load models on first request instead of app start.")

    # ------------------------------------------------------------ thresholds
    score_threshold: float = Field(default=0.30, description="Clamped to [0, 1].")
    mask_threshold: float = Field(default=0.50, description="Clamped to [0, 1].")
    iou_merge_threshold: float = Field(default=0.70, description="Clamped to [0, 1].")
    max_masks: int = Field(default=100, ge=1, le=1000)
    mask_opacity: float = Field(default=0.55, description="Clamped to [0, 1].")

    # ---------------------------------------------------------- annotations
    point_max_size_px: int = Field(default=32, ge=4, description="Max bbox size (px) for a stroke to count as a point.")
    point_max_size_relative: float = Field(
        default=0.05, ge=0.005, le=0.5, description="Max bbox size (fraction of min image dim) for a point."
    )
    cluster_distance_px: int = Field(
        default=48, ge=4, description="Positive points closer than this are grouped into one object."
    )
    negative_point_box_size_relative: float = Field(
        default=0.04,
        ge=0.005,
        le=0.5,
        description=(
            "Mixed mode: size of the negative box derived from a negative point, relative to image width/height."
        ),
    )
    min_component_area: int = Field(default=6, ge=1, description="Ignore annotation strokes smaller than this (px).")
    unknown_color_label: Literal["positive", "negative"] = Field(
        default="positive", description="Sign used when a stroke color cannot be classified as green or red."
    )

    # ------------------------------------------------------------------- ui
    host: str = "0.0.0.0"
    port: int = Field(default=7860, ge=1, le=65535)
    share: bool = False
    gradio_auth: str | None = Field(default=None, description="Optional 'user:password' for the UI.")
    gradio_root_path: str | None = Field(default=None, description="Sub-path if the app is behind a reverse proxy.")
    queue_concurrency: int = Field(default=2, ge=1, le=64)
    max_file_size: str = Field(default="200mb", description="Max upload size accepted by Gradio.")

    # ----------------------------------------------------------------- paths
    output_dir: Path = Field(default=Path("outputs"), description="Where composite images / masks / JSON are saved.")

    # ------------------------------------------------------------- hf hub
    hf_endpoint: str | None = Field(default=None, description="HF endpoint override, e.g. https://hf-mirror.com.")
    hf_token: str | None = Field(default=None, description="HF token for gated/private checkpoints.")

    # ------------------------------------------------------------- misc
    mock: bool = Field(default=False, description="Use the mock engine (no model download). For UI/tests only.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("score_threshold", "mask_threshold", "iou_merge_threshold", "mask_opacity")
    @classmethod
    def _clamp_probabilities(cls, value: float) -> float:
        return _clamp01(value)

    @field_validator("port")
    @classmethod
    def _port_range(cls, value: int) -> int:
        if not (0 < value < 65536):
            raise ValueError("port must be in 1..65535")
        return value

    @property
    def effective_tracker_model_id(self) -> str:
        return self.tracker_model_id or self.model_id

    def apply_hf_environment(self) -> None:
        """Export the configured HF endpoint/token so transformers picks them up."""
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)

    @property
    def auth_tuple(self) -> tuple[str, str] | None:
        if not self.gradio_auth or ":" not in self.gradio_auth:
            return None
        user, _, password = self.gradio_auth.partition(":")
        return (user.strip(), password)

    def model_dump_safe(self) -> dict:
        data = self.model_dump(mode="json")
        if data.get("hf_token"):
            data["hf_token"] = "***"
        return data


def resolve_device(choice: DeviceChoice = DeviceChoice.AUTO) -> str:
    """Resolve 'auto' to an actually available torch device."""
    if choice != DeviceChoice.AUTO:
        return choice.value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def resolve_torch_dtype(choice: DTypeChoice, device: str):
    """Resolve 'auto' to a torch dtype for the given device."""
    import torch

    if choice != DTypeChoice.AUTO:
        return getattr(torch, choice.value)
    if device.startswith(("cuda", "xpu")):
        return torch.bfloat16
    if device == "mps":
        return torch.float32
    return torch.float32


# Re-exported for convenience in docs/tests.
__all__ = [
    "SamSettings",
    "DeviceChoice",
    "DTypeChoice",
    "ModeChoice",
    "resolve_device",
    "resolve_torch_dtype",
]
