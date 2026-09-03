from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sam3_studio.api import create_app
from sam3_studio.engine import MockSegmentationEngine


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(settings, engine=MockSegmentationEngine(settings))
    return TestClient(app)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root_is_api_index(client):
    """The backend serves JSON only — the UI is a separate frontend service."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "sam3-studio-api"
    assert data["segment"] == "/segment"
    assert data["health"] == "/health"
    # No HTML/static from the backend.
    assert "text/html" not in resp.headers.get("content-type", "")


def test_cors_allowlist(client):
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers


def test_health(client, settings):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mock"] is True
    assert data["device"] == "cpu"


def test_config_masks_token(client, settings):
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json()["mock"] is True


def test_segment_text(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={"mode": "text", "text": "car"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["num_instances"] >= 2
    assert data["composite"].startswith("data:image/png;base64,")
    assert data["instances"][0]["mask"].startswith("data:image/png;base64,")
    assert data["files"]["composite"].startswith("/outputs/")
    assert data["files"]["json"].startswith("/outputs/")


def test_segment_points(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={
            "mode": "point",
            "points_positive": json.dumps([[150, 100]]),
            "points_negative": json.dumps([[30, 30]]),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_instances"] >= 1
    assert any("mock" in i["source"] or i["source"] == "point" for i in data["instances"])


def test_segment_boxes_and_mixed(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={
            "mode": "mixed",
            "text": "car",
            "points_positive": json.dumps([[100, 100]]),
            "boxes_negative": json.dumps([[10, 10, 40, 40]]),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["num_instances"] >= 1


def test_segment_no_prompt_422(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={"mode": "auto", "text": ""},
    )
    assert resp.status_code == 422
    assert "Provide a prompt" in resp.json()["detail"]


def test_segment_bad_mode_422(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={"mode": "bogus", "text": "car"},
    )
    assert resp.status_code == 422


def test_segment_bad_points_json_422(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={"mode": "point", "points_positive": "not-json"},
    )
    assert resp.status_code == 422


def test_segment_bad_image_422(client, settings):
    resp = client.post(
        "/segment",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
        data={"mode": "text", "text": "car"},
    )
    assert resp.status_code == 422


def test_segment_no_image_422(client, settings):
    resp = client.post("/segment", data={"mode": "text", "text": "car"})
    assert resp.status_code == 422


def test_outputs_downloadable(client, settings, sample_image):
    resp = client.post(
        "/segment",
        files={"image": ("test.png", _png_bytes(sample_image), "image/png")},
        data={"mode": "text", "text": "car"},
    )
    url = resp.json()["files"]["json"]
    dl = client.get(url)
    assert dl.status_code == 200
    assert dl.json()["num_instances"] >= 2
