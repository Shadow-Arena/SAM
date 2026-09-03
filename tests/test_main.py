from __future__ import annotations

from app.main import parse_args


def test_parse_args_empty_keeps_settings_default():
    """Without flags, lazy_load must stay None so settings decide (default: preload)."""
    args = parse_args([])
    assert args.lazy_load is None


def test_parse_args_preload_flag():
    assert parse_args(["--preload"]).lazy_load is False


def test_parse_args_lazy_flag():
    assert parse_args(["--lazy"]).lazy_load is True


def test_parse_args_config():
    assert parse_args(["--config"]).config is True
