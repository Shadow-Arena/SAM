"""FastAPI application factory (API only — no static/UI serving)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import SamSettings
from ..engine import Engine, get_engine
from .routes import segment_router, system_router


def create_app(settings: SamSettings | None = None, engine: Engine | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``engine`` is optional for tests (inject a mock engine); by default the
    process-wide singleton from :func:`~sam3_studio.engine.get_engine` is used.

    This app is backend-only: it exposes the REST API (``/segment``,
    ``/health``, ``/config``, ``/outputs``) and never serves HTML. The React
    frontend lives in ``frontend/`` and runs separately (Vite dev server, or
    the Nginx container in Docker Compose), talking to this API over HTTP.
    """
    settings = settings or SamSettings()
    engine = engine if engine is not None else get_engine(settings)
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Preload ONCE at startup by default (acceptance criterion).
        if not settings.mock and not settings.lazy_load:
            print("Loading SAM3 model(s) once at startup ...")
            engine.ensure_pcs(print)
            print("SAM3 PCS ready.")
            engine.ensure_tracker(print)
            print(f"SAM3 tracker ready — models cached on {engine.device}.")
        yield
        # End of lifespan: nothing to clean up yet (models stay cached).

    app = FastAPI(
        title="SAM3 Segment Studio API",
        version="0.4.0",
        description=(
            "Backend API for SAM3 Segment Studio — interactive segmentation "
            "with text, box, point or mixed prompts. Powered by facebook/sam3 "
            "via 🤗 Transformers."
        ),
        lifespan=lifespan,
    )

    # Allow the separately-hosted frontend (Vite dev server, Nginx container,
    # or any deployment origin) to call this API cross-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")
    app.state.settings = settings
    app.state.engine = engine
    app.state.output_dir = output_dir
    app.include_router(system_router)
    app.include_router(segment_router)
    return app


__all__ = ["create_app"]
