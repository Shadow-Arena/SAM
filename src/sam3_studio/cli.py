"""One-shot segmentation from the command line.

Examples:
    python -m sam3_studio.cli --image image.jpg --text "car"
    python -m sam3_studio.cli --image image.jpg --point 320 240 --point 500 400 --negative-point 100 100
    python -m sam3_studio.cli --image image.jpg --box 100 150 500 450
    python -m sam3_studio.cli --image image.jpg --text "handle" --negative-box 40 183 318 204 --mode mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .config import ModeChoice, SamSettings
from .domain import XYXY, PromptSet
from .engine import SegmentationError, create_engine
from .export import save_result
from .rendering import overlay_masks


def _pair(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'x,y' got {value!r}")
    return (int(parts[0]), int(parts[1]))


def _box(value: str) -> XYXY:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected 'x1,y1,x2,y2' got {value!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment an image with SAM3 (text/box/point/mixed)")
    parser.add_argument("--image", required=True, help="Input image path or URL.")
    parser.add_argument("--text", default=None, help="Text concept prompt, e.g. 'yellow school bus'.")
    parser.add_argument(
        "--point", action="append", type=_pair, default=[], metavar="X,Y", help="Positive point (repeatable)."
    )
    parser.add_argument(
        "--negative-point", action="append", type=_pair, default=[], metavar="X,Y", help="Negative point (repeatable)."
    )
    parser.add_argument(
        "--box", action="append", type=_box, default=[], metavar="X1,Y1,X2,Y2", help="Positive box (repeatable)."
    )
    parser.add_argument(
        "--negative-box",
        action="append",
        type=_box,
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="Negative box (repeatable).",
    )
    parser.add_argument("--mode", choices=[m.value for m in ModeChoice], default="auto")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--max-masks", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mock", action="store_true", help="Use the mock engine (no model).")
    parser.add_argument("--show", action="store_true", help="Open the composite image with the default viewer.")
    return parser.parse_args(argv)


def load_image(source: str) -> Image.Image:
    """Load a local image file or fetch a URL."""
    from io import BytesIO
    from urllib.parse import urlparse

    path = Path(source)
    if path.exists():
        return Image.open(path).convert("RGB")
    if not urlparse(source).scheme:
        raise FileNotFoundError(f"Image not found: {source}")
    import requests

    resp = requests.get(source, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def _is_path(source: str) -> bool:
    return Path(source).exists()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = SamSettings()
    updates: dict = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.mock:
        updates["mock"] = True
    if updates:
        settings = settings.model_copy(update=updates)

    prompt = PromptSet(
        text=args.text,
        points_positive=list(args.point),
        points_negative=list(args.negative_point),
        boxes_positive=list(args.box),
        boxes_negative=list(args.negative_box),
    )
    if not prompt:
        print("error: no prompts given (use --text/--point/--box/--negative-*).", file=sys.stderr)
        return 2

    image = load_image(args.image)
    engine = create_engine(settings)
    try:
        result = engine.segment(
            image,
            prompt,
            mode=args.mode,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            max_masks=args.max_masks,
            progress=print,
        )
    except SegmentationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    composite = overlay_masks(image, result.instances)
    run_id = f"cli_{Path(args.image).stem if _is_path(args.image) else 'image'}_{id(result) % 10000}"
    paths = save_result(composite, result, settings.output_dir, run_id)

    print(f"Image size: {result.image_size[1]}x{result.image_size[0]}")
    print(f"Found {len(result.instances)} object(s) in {result.elapsed_seconds:.2f}s")
    for inst in result.instances:
        score = f"{inst.score:.3f}" if inst.score is not None else "n/a"
        print(f"  #{inst.object_id} [{inst.source}] score={score} box={inst.box} area={inst.area_px}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print(f"Saved: {paths['composite']}")
    print(f"       {paths['json']}")
    if args.show:
        composite.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
