"""System routes: API index, status and configuration."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...config import SamSettings
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> JSONResponse:
    """API index — the backend serves JSON only (frontend runs separately)."""
    return JSONResponse(
        {
            "service": "sam3-studio-api",
            "version": "0.4.0",
            "docs": "/docs",
            "health": "/health",
            "config": "/config",
            "segment": "/segment",
        }
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Engine / model status."""
    settings: SamSettings = request.app.state.settings
    engine = request.app.state.engine
    loaded = bool(getattr(engine, "loaded", False)) or settings.mock
    return HealthResponse(
        status="ok",
        mock=settings.mock,
        device=getattr(engine, "device", "unknown"),
        model_loaded=loaded,
        lazy_load=settings.lazy_load,
        model_id=settings.model_id,
        hf_auth=settings.describe_hf_auth(),
    )


@router.get("/config")
def config_public(request: Request) -> dict:
    """Effective configuration (token masked)."""
    return request.app.state.settings.model_dump_safe()
