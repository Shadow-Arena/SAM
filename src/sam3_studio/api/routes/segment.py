"""``POST /segment`` — run a segmentation request."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from ...config import ModeChoice
from ...domain import PromptSet
from ...engine import SegmentationError
from ...export import SaveResultPaths, save_result
from ...rendering import overlay_masks, overlay_semantic
from ..deps import load_image, parse_boxes, parse_points, png_data_uri
from ..schemas import SegmentResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Models are heavy and not safely re-entrant during a single forward pass.
INFERENCE_LOCK = threading.Lock()


def _output_paths_to_urls(paths: SaveResultPaths, output_dir: Path) -> dict:
    """Turn saved result paths into URLs under the static /outputs mount."""
    root = output_dir.resolve()
    urls: dict = {}
    for key, value in paths.items():
        if key == "masks":
            urls[key] = ["/outputs/" + Path(p).resolve().relative_to(root).as_posix() for p in value]
        else:
            urls[key] = "/outputs/" + Path(value).resolve().relative_to(root).as_posix()
    return urls


@router.post("/segment", response_model=SegmentResponse)
def segment(
    request: Request,
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
) -> SegmentResponse:
    """Run one segmentation request and return composite + masks as data URIs."""
    engine = request.app.state.engine
    output_dir: Path = request.app.state.output_dir

    img = load_image(image)
    prompt = PromptSet(
        text=text.strip() or None,
        points_positive=parse_points(points_positive, "points_positive"),
        points_negative=parse_points(points_negative, "points_negative"),
        boxes_positive=parse_boxes(boxes_positive, "boxes_positive"),
        boxes_negative=parse_boxes(boxes_negative, "boxes_negative"),
    )
    if not prompt:
        raise HTTPException(status_code=422, detail="Provide a prompt: text, point(s) and/or box(es).")
    if mode.lower() not in ModeChoice._value2member_map_:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(ModeChoice)}")
    mode_enum = ModeChoice(mode.lower())

    try:
        with INFERENCE_LOCK:
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
        record["mask"] = png_data_uri(mask_img)
        instances.append(record)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "mode": mode_enum.value,
        "prompt": prompt.describe(),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "num_instances": len(result.instances),
        "instances": instances,
        "composite": png_data_uri(composite),
        "semantic": png_data_uri(Image.fromarray(result.semantic_mask.astype(np.uint8) * 255))
        if result.semantic_mask is not None
        else None,
        "warnings": result.warnings,
        "files": _output_paths_to_urls(paths, output_dir),
    }
    return payload
