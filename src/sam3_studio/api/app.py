"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import SamSettings
from ..engine import Engine, get_engine
from .routes import segment_router, system_router


def create_app(settings: SamSettings | None = None, engine: Engine | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``engine`` is optional for tests (inject a mock engine); by default the
    process-wide singleton from :func:`~sam3_studio.engine.get_engine` is used.
    """
    settings = settings or SamSettings()
    engine = engine if engine is not None else get_engine(settings)
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_dir = Path(__file__).resolve().parent.parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

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
        title="SAM3 Segment Studio",
        version="0.2.0",
        description=(
            "Interactive SAM3 segmentation: text, box, point or mixed prompts. "
            "Powered by facebook/sam3 via 🤗 Transformers."
        ),
        lifespan=lifespan,
    )
    app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.state.settings = settings
    app.state.engine = engine
    app.state.output_dir = output_dir
    app.include_router(system_router)
    app.include_router(segment_router)
    return app


__all__ = ["create_app"]
