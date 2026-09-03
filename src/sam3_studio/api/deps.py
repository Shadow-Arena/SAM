"""FastAPI dependencies and request parsing helpers."""

from __future__ import annotations

import base64
import io
import json

from fastapi import HTTPException, UploadFile
from PIL import Image


def parse_points(raw: str, field: str) -> list[tuple[int, int]]:
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


def parse_boxes(raw: str, field: str) -> list[tuple[int, int, int, int]]:
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


def load_image(upload: UploadFile) -> Image.Image:
    """Decode an uploaded file into an RGB PIL image (422 on failure)."""
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=422, detail="image: empty upload")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL raises several types
        raise HTTPException(status_code=422, detail="image: could not decode the uploaded file") from exc


def png_data_uri(image: Image.Image) -> str:
    """Encode an image as an inline ``data:image/png;base64,...`` URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
