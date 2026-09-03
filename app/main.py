"""Entry point: launch the interactive Gradio app.

Usage:
    python -m app.main [--host 0.0.0.0] [--port 7860] [--share] [--mock]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import SamSettings
from .ui import build_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 Segment Studio — interactive image segmentation")
    parser.add_argument("--host", default=None, help="Server bind address (default: settings.host).")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: settings.port).")
    parser.add_argument("--share", action="store_true", default=None, help="Create a public Gradio share link.")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable public share link.")
    parser.add_argument("--mock", action="store_true", default=None, help="Run with the mock engine (no model).")
    parser.add_argument(
        "--lazy",
        dest="lazy_load",
        action="store_true",
        default=None,
        help="Skip the startup model load; load on the first request instead.",
    )
    parser.add_argument("--config", action="store_true", help="Print effective configuration and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = SamSettings()
    updates: dict = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.share is not None:
        updates["share"] = args.share
    if args.mock is not None:
        updates["mock"] = args.mock
    if args.lazy_load is not None:
        updates["lazy_load"] = args.lazy_load
    if updates:
        settings = settings.model_copy(update=updates)

    logging.basicConfig(
        level=getattr(logging, settings.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.config:
        for key, value in sorted(settings.model_dump_safe().items()):
            print(f"{key:32s} = {value}")
        print(f"{'hf_auth':32s} = {settings.describe_hf_auth()}")
        return 0

    import gradio as gr

    demo = build_app(settings)
    # Load the model ONCE, synchronously at startup, so the UI is ready before
    # the first segmentation request (requested default behavior).
    if not settings.mock and not settings.lazy_load:
        from .segmentation import get_engine

        print("Loading SAM3 model(s) once at startup ...")
        engine = get_engine(settings)
        engine.ensure_pcs(print)
        print("SAM3 PCS ready.")
        engine.ensure_tracker(print)
        print(f"SAM3 tracker ready — models cached on {engine.device}.")

    demo.queue(default_concurrency_limit=settings.queue_concurrency)
    demo.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=settings.share,
        auth=settings.auth_tuple,
        root_path=settings.gradio_root_path,
        max_file_size=settings.max_file_size,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        css="footer {display: none !important;}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
