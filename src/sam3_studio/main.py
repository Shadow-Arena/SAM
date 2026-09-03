"""Entry point: launch the FastAPI API server.

Usage:
    python -m sam3_studio.main [--host 0.0.0.0] [--port 8000] [--mock] [--lazy] [--reload]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .api import create_app
from .config import SamSettings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 Segment Studio — FastAPI server")
    parser.add_argument("--host", default=None, help="Server bind address (default: settings.host).")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: settings.port).")
    parser.add_argument("--mock", action="store_true", default=None, help="Run with the mock engine (no model).")
    parser.add_argument(
        "--lazy",
        dest="lazy_load",
        action="store_true",
        default=None,
        help="Skip the startup model load; load on the first request instead.",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (development).")
    parser.add_argument("--config", action="store_true", help="Print effective configuration and exit.")
    return parser.parse_args(argv)


def _apply_overrides(settings: SamSettings, args: argparse.Namespace) -> SamSettings:
    updates: dict = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.mock is not None:
        updates["mock"] = args.mock
    if args.lazy_load is not None:
        updates["lazy_load"] = args.lazy_load
    return settings.model_copy(update=updates) if updates else settings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = _apply_overrides(SamSettings(), args)

    logging.basicConfig(
        level=getattr(logging, settings.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.config:
        for key, value in sorted(settings.model_dump_safe().items()):
            print(f"{key:32s} = {value}")
        print(f"{'hf_auth':32s} = {settings.describe_hf_auth()}")
        return 0

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower(), reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
