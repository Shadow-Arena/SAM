from __future__ import annotations

import os

import pytest

from app import cli


def test_cli_text_mode(sample_image, tmp_path, monkeypatch):
    img_path = tmp_path / "in.png"
    sample_image.save(img_path)
    out = tmp_path / "out"
    rc = cli.main(
        [
            "--image",
            str(img_path),
            "--text",
            "car",
            "--mock",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    files = list(out.rglob("*.json"))
    assert len(files) == 1
    assert os.path.exists(str(out / files[0].parent / "composite.png"))


def test_cli_point_mode(sample_image, tmp_path, monkeypatch):
    img_path = tmp_path / "in.png"
    sample_image.save(img_path)
    out = tmp_path / "out2"
    rc = cli.main(
        [
            "--image",
            str(img_path),
            "--point",
            "150,100",
            "--negative-point",
            "30,30",
            "--mode",
            "point",
            "--mock",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0


def test_cli_no_prompt_returns_2(sample_image, tmp_path):
    img_path = tmp_path / "in.png"
    sample_image.save(img_path)
    rc = cli.main(["--image", str(img_path), "--mock"])
    assert rc == 2


def test_cli_load_image_url_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.load_image(str(tmp_path / "does-not-exist.png"))
