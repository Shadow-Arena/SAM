from __future__ import annotations

from app.main import parse_args


def test_parse_args_defaults_keep_settings():
    """Without flags, lazy_load stays None so settings decide (default: preload at start)."""
    args = parse_args([])
    assert args.lazy_load is None
    assert args.share is None
    assert args.mock is None


def test_parse_args_lazy_flag():
    assert parse_args(["--lazy"]).lazy_load is True


def test_parse_args_config():
    assert parse_args(["--config"]).config is True
