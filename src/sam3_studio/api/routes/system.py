"""System routes: status and configuration."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from ...config import SamSettings
from ..schemas import HealthResponse

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-file web UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


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
