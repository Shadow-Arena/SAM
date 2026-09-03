"""Pydantic response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InstanceOut(BaseModel):
    """A single detected object."""

    id: int
    label: str = ""
    source: str
    score: float | None = None
    box: list[int]
    area_px: int
    mask: str = Field(description="Inline PNG data URI of the mask.")


class FilesOut(BaseModel):
    """Saved-file URLs under the /outputs mount."""

    model_config = ConfigDict(populate_by_name=True)

    composite: str | None = None
    json_url: str | None = Field(default=None, alias="json")
    semantic: str | None = None
    masks: list[str] = Field(default_factory=list)


class SegmentResponse(BaseModel):
    """Response of ``POST /segment``."""

    status: str = "ok"
    run_id: str
    mode: str
    prompt: str
    elapsed_seconds: float
    num_instances: int
    instances: list[InstanceOut]
    composite: str = Field(description="Inline PNG data URI of the annotated image.")
    semantic: str | None = None
    warnings: list[str] = Field(default_factory=list)
    files: FilesOut


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str
    mock: bool
    device: str
    model_loaded: bool
    lazy_load: bool
    model_id: str
    hf_auth: str
