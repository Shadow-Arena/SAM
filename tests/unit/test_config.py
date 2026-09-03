from __future__ import annotations

import os

from sam3_studio.config import DeviceChoice, DTypeChoice, SamSettings, resolve_device, resolve_torch_dtype


def test_defaults():
    s = SamSettings(_env_file=None)
    assert s.model_id == "facebook/sam3"
    assert s.effective_tracker_model_id == "facebook/sam3"
    assert s.score_threshold == 0.30
    assert s.port == 8000  # API port; the UI runs on its own port (7860)
    assert "localhost:5173" in s.cors_origins
    assert s.hf_token is None
    assert s.lazy_load is False  # model loads once at startup by default


def test_tracker_fallback():
    s = SamSettings(tracker_model_id="facebook/sam3-large", _env_file=None)
    assert s.effective_tracker_model_id == "facebook/sam3-large"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SAM_MODEL_ID", "facebook/sam3-base")
    monkeypatch.setenv("SAM_PORT", "8000")
    monkeypatch.setenv("SAM_SCORE_THRESHOLD", "0.9")
    s = SamSettings(_env_file=None)
    assert s.model_id == "facebook/sam3-base"
    assert s.port == 8000
    assert abs(s.score_threshold - 0.9) < 1e-9


def test_thresholds_clamped():
    s = SamSettings(score_threshold=2.0, mask_threshold=-1.0, _env_file=None)
    assert s.score_threshold == 1.0
    assert s.mask_threshold == 0.0


def test_model_dump_masks_token():
    s = SamSettings(hf_token="hf_super_secret", _env_file=None)
    assert s.model_dump_safe()["hf_token"] == "***"


def test_without_env_file():
    s = SamSettings(_env_file=None)
    assert s.host == "0.0.0.0"


# ------------------------------------------------------------- hf login
def test_single_login_var_both_sources(monkeypatch):
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_token_from_env")
    s = SamSettings(_env_file=None)
    assert s.hf_token == "hf_token_from_env"


def test_single_login_var_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SAM_HF_TOKEN=hf_token_from_dotenv\n", encoding="utf-8")
    s = SamSettings(_env_file=str(env_file))
    assert s.hf_token == "hf_token_from_dotenv"


def test_plain_hf_token_does_not_override_sam_token(monkeypatch):
    """HF_TOKEN alone is NOT a login var anymore; SAM_HF_TOKEN is the only one."""
    monkeypatch.setenv("HF_TOKEN", "hf_ambient")
    monkeypatch.delenv("SAM_HF_TOKEN", raising=False)
    s = SamSettings(_env_file=None)
    assert s.hf_token is None


def test_apply_hf_environment_sets_env(monkeypatch):
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()
    assert os.environ.get("HF_TOKEN") == "hf_test_token"
    assert os.environ.get("HUGGINGFACEHUB_API_TOKEN") == "hf_test_token"


def test_apply_hf_environment_logs_in(monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "login", lambda token=None, **kw: calls.append((token, kw)))
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()
    assert calls == [("hf_test_token", {"add_to_git_credential": False})]


def test_apply_hf_environment_login_failure_does_not_raise(monkeypatch):
    import huggingface_hub

    def _offline_login(token=None, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(huggingface_hub, "login", _offline_login)
    monkeypatch.setenv("SAM_HF_TOKEN", "hf_test_token")
    s = SamSettings(_env_file=None)
    s.apply_hf_environment()  # must not raise
    assert os.environ.get("HF_TOKEN") == "hf_test_token"


def test_describe_hf_auth_hides_token():
    assert SamSettings(_env_file=None).describe_hf_auth() == "not configured (set SAM_HF_TOKEN in .env)"
    desc = SamSettings(hf_token="hf_very_secret_token", _env_file=None).describe_hf_auth()
    assert "very_secret" not in desc
    assert "oken" in desc


def test_no_mirror_settings():
    """Mirror/endpoint settings are gone — SAM_HF_TOKEN is the only HF setting."""
    s = SamSettings(_env_file=None)
    assert not hasattr(s, "hf_endpoint")
    assert not hasattr(s, "hf_login")


def test_no_gradio_settings():
    """Gradio-specific settings are gone in the FastAPI version."""
    s = SamSettings(_env_file=None)
    for name in ("share", "gradio_auth", "gradio_root_path", "queue_concurrency", "max_file_size"):
        assert not hasattr(s, name)


def test_resolve_device_types():
    assert resolve_device(DeviceChoice.CPU) == "cpu"
    assert resolve_device(DeviceChoice.AUTO) in {"cpu", "cuda", "mps", "xpu"}


def test_resolve_dtype_auto_cpu():
    import torch

    assert resolve_torch_dtype(DTypeChoice.AUTO, "cpu") == torch.float32
    assert resolve_torch_dtype(DTypeChoice.BFLOAT16, "cpu") == torch.bfloat16
