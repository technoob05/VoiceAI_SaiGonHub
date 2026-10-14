"""Privacy defaults for the production supervisor."""

from __future__ import annotations

import sys
from pathlib import Path


DEVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEVICE_DIR))

import device_config  # noqa: E402
import devlog  # noqa: E402
import supervisor  # noqa: E402


def _supervisor(monkeypatch, enabled: bool = False):
    monkeypatch.setenv("VOICEAI_LOG_CONTENT", "1" if enabled else "0")
    return supervisor.Supervisor(log=devlog.Logger("privacy-test", console=False))


def test_content_logging_requires_explicit_opt_in():
    assert supervisor.content_logging_enabled({}) is False
    assert supervisor.content_logging_enabled({"VOICEAI_LOG_CONTENT": "0"}) is False
    assert supervisor.content_logging_enabled({"VOICEAI_LOG_CONTENT": "1"}) is True
    assert device_config.validate({"VOICEAI_LOG_CONTENT": "true"})[0]


def test_mic_direction_is_validated_before_model_startup():
    check = supervisor.mic_configuration_error
    assert check(mic=True, pair="vi,en", source=None,
                 multilingual_asr=True, target="en")
    assert check(mic=True, pair="vi,en", source="en",
                 multilingual_asr=False, target="en")
    assert check(mic=True, pair="vi,en", source="ko",
                 multilingual_asr=True, target="en")
    assert check(mic=True, pair="en,ko", source="en",
                 multilingual_asr=True, target="en")
    assert check(mic=True, pair="vi,en", source="vi",
                 multilingual_asr=False, target="en") is None
    assert check(mic=True, pair="vi,en", source="en",
                 multilingual_asr=True, target="en") is None


def test_mic_audit_fields_are_metadata_only_by_default(monkeypatch):
    sup = _supervisor(monkeypatch)
    sup.mic["turns"] = 3
    fields = sup._mic_turn_log_fields({
        "source": "vi",
        "target": "en",
        "src": "Mật khẩu nhà máy là 1234",
        "tgt": "The factory password is 1234",
        "total_ms": 451.2,
    })
    assert fields == {
        "turn": 3,
        "source": "vi",
        "target": "en",
        "src_chars": 24,
        "tgt_chars": 28,
        "total_ms": 451.2,
    }
    assert sup.health()["privacy"] == {
        "content_logging": False,
        "structured_log_policy": "metadata-only",
    }


def test_content_preview_is_bounded_when_operator_opts_in(monkeypatch):
    sup = _supervisor(monkeypatch, enabled=True)
    fields = sup._mic_turn_log_fields({"src": "x" * 200, "tgt": "y" * 200})
    assert fields["src_preview"] == "x" * 120
    assert fields["tgt_preview"] == "y" * 120
    assert sup.health()["privacy"]["content_logging"] is True
