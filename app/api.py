"""FastAPI application exposing the SAM3 segmentation engine.

Endpoints:

* ``GET  /``        — the interactive web UI (single-file HTML/JS page)
* ``GET  /health``  — engine / model status
* ``GET  /config``  — effective server configuration (token masked)
* ``POST /segment`` — run a segmentation (multipart: image + prompts)

The engine is created once per process. Unless ``SAM_LAZY_LOAD=true``, the
SAM3 models are loaded synchronously at startup (FastAPI lifespan), so the
first request is fast.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import ModeChoice, SamSettings
from .schemas import PromptSet
from .segmentation import SegmentationError, get_engine
from .visualization import overlay_masks, overlay_semantic, save_result

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Models are heavy and not safely re-entrant during a single forward pass; the
# old Gradio handler used the same kind of lock.
_INFERENCE_LOCK = threading.Lock()


# --------------------------------------------------------------------- helpers
def _parse_points(raw: str, field: str) -> list[tuple[int, int]]:
    """Parse a JSON array of ``[x, y]`` pairs from a multipart string field."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field}: invalid JSON list") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=422, detail=f"{field}: expected a JSON list")
    points: list[tuple[int, int]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise HTTPException(status_code=422, detail=f"{field}: each item must be [x, y]")
        try:
            points.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{field}: coordinates must be numbers") from exc
    return points


def _parse_boxes(raw: str, field: str) -> list[tuple[int, int, int, int]]:
    """Parse a JSON array of ``[x1, y1, x2, y2]`` boxes from a multipart string field."""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"{field}: invalid JSON list") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=422, detail=f"{field}: expected a JSON list")
    boxes: list[tuple[int, int, int, int]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise HTTPException(status_code=422, detail=f"{field}: each item must be [x1, y1, x2, y2]")
        try:
            boxes.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{field}: coordinates must be numbers") from exc
    return boxes


def _load_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=422, detail="image: empty upload")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises several types
        raise HTTPException(status_code=422, detail="image: could not decode the uploaded file") from exc


def _png_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _output_paths_to_urls(paths: dict[str, str], output_dir: Path) -> dict:
    """Turn saved result paths into URLs under the static /outputs mount."""
    root = output_dir.resolve()
    urls: dict = {}
    for key, value in paths.items():
        if key == "masks":
            urls[key] = ["/outputs/" + Path(p).resolve().relative_to(root).as_posix() for p in value]
        else:
            urls[key] = "/outputs/" + Path(value).resolve().relative_to(root).as_posix()
    return urls


# ------------------------------------------------------------------- app factory
def create_app(settings: SamSettings | None = None, engine=None) -> FastAPI:
    """Build the FastAPI application.

    ``engine`` is optional for tests (inject a mock engine); by default the
    process-wide singleton from :func:`app.segmentation.get_engine` is used.
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
    app.state.settings = settings
    app.state.engine = engine

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    # ---------------------------------------------------------------- status
    @app.get("/health")
    def health() -> dict:
        loaded = bool(getattr(engine, "loaded", False)) or settings.mock
        return {
            "status": "ok",
            "mock": settings.mock,
            "device": getattr(engine, "device", "unknown"),
            "model_loaded": loaded,
            "lazy_load": settings.lazy_load,
            "model_id": settings.model_id,
            "hf_auth": settings.describe_hf_auth(),
        }

    @app.get("/config")
    def config_public() -> dict:
        return settings.model_dump_safe()

    # -------------------------------------------------------------- segment
    @app.post("/segment")
    def segment(
        image: UploadFile = File(..., description="Input image (PNG/JPEG/WebP)."),
        mode: str = Form("auto", description="auto | text | box | point | mixed"),
        text: str = Form("", description="Text concept prompt (PCS)."),
        points_positive: str = Form("[]", description="JSON list of [x, y] positive clicks."),
        points_negative: str = Form("[]", description="JSON list of [x, y] negative clicks."),
        boxes_positive: str = Form("[]", description="JSON list of [x1, y1, x2, y2] positive boxes."),
        boxes_negative: str = Form("[]", description="JSON list of [x1, y1, x2, y2] negative boxes."),
        score_threshold: float | None = Form(None),
        mask_threshold: float | None = Form(None),
        max_masks: int | None = Form(None),
        opacity: float = Form(0.55, ge=0.0, le=1.0),
        show_semantic: bool = Form(False),
    ) -> JSONResponse:
        """Run one segmentation request and return composite + masks as data URIs."""
        img = _load_image(image)
        prompt = PromptSet(
            text=text.strip() or None,
            points_positive=_parse_points(points_positive, "points_positive"),
            points_negative=_parse_points(points_negative, "points_negative"),
            boxes_positive=_parse_boxes(boxes_positive, "boxes_positive"),
            boxes_negative=_parse_boxes(boxes_negative, "boxes_negative"),
        )
        if not prompt:
            raise HTTPException(status_code=422, detail="Provide a prompt: text, point(s) and/or box(es).")
        if mode.lower() not in ModeChoice._value2member_map_:
            raise HTTPException(status_code=422, detail=f"mode must be one of {list(ModeChoice)}")
        mode_enum = ModeChoice(mode.lower())

        try:
            with _INFERENCE_LOCK:
                result = engine.segment(
                    img,
                    prompt,
                    mode=mode_enum,
                    score_threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    max_masks=max_masks,
                )
        except SegmentationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        composite = overlay_masks(img, result.instances, opacity=opacity, draw_boxes=True)
        if show_semantic and result.semantic_mask is not None:
            composite = overlay_semantic(composite, result.semantic_mask, opacity=0.35)

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        paths = save_result(composite, result, output_dir, run_id)

        instances = []
        for inst in result.instances:
            mask_img = Image.fromarray(inst.mask.astype(np.uint8) * 255)
            record = inst.to_record()
            record["mask"] = _png_data_uri(mask_img)
            instances.append(record)

        payload = {
            "status": "ok",
            "run_id": run_id,
            "mode": mode_enum.value,
            "prompt": prompt.describe(),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "num_instances": len(result.instances),
            "instances": instances,
            "composite": _png_data_uri(composite),
            "semantic": _png_data_uri(Image.fromarray(result.semantic_mask.astype(np.uint8) * 255))
            if result.semantic_mask is not None
            else None,
            "warnings": result.warnings,
            "files": _output_paths_to_urls(paths, output_dir),
        }
        return JSONResponse(payload)

    return app


def main_app() -> FastAPI:
    """ASGI entry point (used by ``uvicorn app.api:main_app``)."""
    return create_app()


__all__ = ["create_app", "main_app"]
