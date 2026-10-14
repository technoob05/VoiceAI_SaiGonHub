"""Tests for asr_router — routing, laziness, residency and language forcing.

None of these tests load a real Whisper model. `AsrRouter` takes an injectable
loader, so a fake model factory records exactly what the router asked for: which
directory it loaded, how many times it constructed a model, and which language
token it forced on each decode. That is the whole contract worth testing, and it
runs in milliseconds on a machine with no weights at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asr_router
from asr_router import (
    ROLE_MULTI,
    ROLE_VI,
    SAMPLE_RATE,
    AsrRouter,
    AsrRouterConfig,
    TranscriptionResult,
    load_wav_mono16k,
    normalize_lang,
    resample_to_16k,
)


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeInfo:
    language_probability = 0.99


class FakeModel:
    """Stands in for faster_whisper.WhisperModel and records every call."""

    def __init__(self, model_dir, cfg):
        self.model_dir = model_dir
        self.cfg = cfg
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"n_samples": len(audio), **kwargs})
        tag = Path(self.model_dir).name
        return iter([FakeSegment(f"[{tag}:{kwargs.get('language')}]")]), FakeInfo()


class FakeFactory:
    """Counts constructions so laziness and residency are directly observable."""

    def __init__(self):
        self.built = []

    def __call__(self, model_dir, cfg):
        model = FakeModel(model_dir, cfg)
        self.built.append(model)
        return model

    @property
    def dirs(self):
        return [m.model_dir for m in self.built]


@pytest.fixture
def model_dirs(tmp_path):
    """Two directories that exist and carry a little weight, so footprint is non-zero."""
    vi = tmp_path / "phowhisper-small-int8"
    multi = tmp_path / "whisper-small-multilingual"
    for d, size in ((vi, 2048), (multi, 4096)):
        d.mkdir()
        (d / "model.bin").write_bytes(b"\0" * size)
    return vi, multi


@pytest.fixture
def factory():
    return FakeFactory()


@pytest.fixture
def router(model_dirs, factory):
    vi, multi = model_dirs
    cfg = AsrRouterConfig(vi_model_dir=str(vi), multi_model_dir=str(multi))
    return AsrRouter(config=cfg, loader=factory, verbose=False)


def audio(seconds=1.0, sr=SAMPLE_RATE):
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------- routing

def test_vietnamese_routes_to_the_specialist(router, model_dirs):
    vi_dir, _ = model_dirs
    result = router.transcribe(audio(), "vi")
    assert result.role == ROLE_VI
    assert result.model_path == str(vi_dir)


@pytest.mark.parametrize("lang", ["en", "zh", "ko"])
def test_non_vietnamese_routes_to_the_multilingual_model(router, model_dirs, lang):
    _, multi_dir = model_dirs
    result = router.transcribe(audio(), lang)
    assert result.role == ROLE_MULTI
    assert result.model_path == str(multi_dir)


def test_role_for_is_a_pure_lookup():
    assert AsrRouter.role_for("vi") == ROLE_VI
    assert AsrRouter.role_for("VI ") == ROLE_VI
    for lang in ("en", "zh", "ko"):
        assert AsrRouter.role_for(lang) == ROLE_MULTI


def test_unsupported_language_is_rejected(router):
    with pytest.raises(ValueError):
        router.transcribe(audio(), "ja")
    with pytest.raises(ValueError):
        normalize_lang("")


# ---------------------------------------------------------------- laziness

def test_vietnamese_only_session_never_builds_the_multilingual_model(router, factory, model_dirs):
    vi_dir, multi_dir = model_dirs
    for _ in range(3):
        router.transcribe(audio(), "vi")
    assert factory.dirs == [str(vi_dir)]
    assert str(multi_dir) not in factory.dirs
    assert not router.is_loaded(ROLE_MULTI)
    assert router.loaded_roles() == (ROLE_VI,)


def test_nothing_is_loaded_before_the_first_call(router):
    assert router.loaded_roles() == ()
    assert not router.is_loaded(ROLE_VI)
    assert not router.is_loaded(ROLE_MULTI)


def test_multilingual_call_does_not_drag_in_the_specialist(router, factory, model_dirs):
    _, multi_dir = model_dirs
    router.transcribe(audio(), "ko")
    assert factory.dirs == [str(multi_dir)]
    assert not router.is_loaded(ROLE_VI)


# ---------------------------------------------------------------- residency

def test_second_call_reuses_the_same_instance(router, factory):
    router.transcribe(audio(), "vi")
    router.transcribe(audio(), "vi")
    router.transcribe(audio(), "vi")
    assert len(factory.built) == 1
    assert len(factory.built[0].calls) == 3


def test_alternating_languages_build_each_model_exactly_once(router, factory):
    for lang in ("vi", "en", "vi", "zh", "ko", "vi"):
        router.transcribe(audio(), lang)
    assert len(factory.built) == 2
    vi_model = next(m for m in factory.built if "phowhisper" in m.model_dir)
    multi_model = next(m for m in factory.built if "multilingual" in m.model_dir)
    assert len(vi_model.calls) == 3
    assert len(multi_model.calls) == 3


def test_model_for_returns_the_resident_object(router):
    first = router.model_for("en")
    assert router.model_for("zh") is first
    assert router.model_for("ko") is first
    assert router.model_for("vi") is not first


def test_load_ms_is_charged_only_to_the_first_call(router):
    first = router.transcribe(audio(), "vi")
    second = router.transcribe(audio(), "vi")
    assert first.load_ms > 0.0
    assert second.load_ms == 0.0


# ---------------------------------------------------------------- language forcing

@pytest.mark.parametrize("lang", ["vi", "en", "zh", "ko"])
def test_language_token_is_always_forced(router, factory, lang):
    result = router.transcribe(audio(), lang)
    model = factory.built[-1]
    assert model.calls[-1]["language"] == lang
    assert model.calls[-1]["task"] == "transcribe"
    assert result.forced_language == lang
    assert result.lang == lang


def test_vietnamese_prompt_never_leaks_into_the_multilingual_decoder(model_dirs, factory):
    vi_dir, multi_dir = model_dirs
    cfg = AsrRouterConfig(
        vi_model_dir=str(vi_dir),
        multi_model_dir=str(multi_dir),
        vi_initial_prompt="máy tính, điện thoại",
    )
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    router.transcribe(audio(), "vi")
    router.transcribe(audio(), "en")
    vi_model = next(m for m in factory.built if m.model_dir == str(vi_dir))
    multi_model = next(m for m in factory.built if m.model_dir == str(multi_dir))
    assert vi_model.calls[-1]["initial_prompt"] == "máy tính, điện thoại"
    assert "initial_prompt" not in multi_model.calls[-1]


def test_decode_settings_match_the_existing_pipeline(router, factory):
    router.transcribe(audio(), "vi")
    call = factory.built[0].calls[-1]
    assert call["beam_size"] == 1
    assert call["temperature"] == 0.0
    assert call["condition_on_previous_text"] is False
    assert call["without_timestamps"] is True
    assert call["no_repeat_ngram_size"] == 3
    assert call["vad_parameters"] == {"min_silence_duration_ms": 500}


# ---------------------------------------------------------------- missing model

def test_missing_model_path_raises_with_a_recovery_hint(tmp_path, factory):
    cfg = AsrRouterConfig(
        vi_model_dir=str(tmp_path / "nope-vi"),
        multi_model_dir=str(tmp_path / "nope-multi"),
    )
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    with pytest.raises(FileNotFoundError) as exc:
        router.transcribe(audio(), "en")
    assert "snapshot_download" in str(exc.value)
    assert factory.built == []


def test_missing_multilingual_does_not_break_the_vietnamese_path(model_dirs, tmp_path, factory):
    vi_dir, _ = model_dirs
    cfg = AsrRouterConfig(vi_model_dir=str(vi_dir), multi_model_dir=str(tmp_path / "absent"))
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    assert router.transcribe(audio(), "vi").role == ROLE_VI
    with pytest.raises(FileNotFoundError):
        router.transcribe(audio(), "zh")


def test_missing_vietnamese_model_names_the_converter(model_dirs, tmp_path, factory):
    _, multi_dir = model_dirs
    cfg = AsrRouterConfig(vi_model_dir=str(tmp_path / "absent"), multi_model_dir=str(multi_dir))
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    with pytest.raises(FileNotFoundError) as exc:
        router.transcribe(audio(), "vi")
    assert "ctranslate2.converters" in str(exc.value)


# ---------------------------------------------------------------- prewarm

def test_prewarm_pair_loads_both_models_and_runs_one_dummy_each(router, factory):
    report = router.prewarm_pair(("vi", "en"))
    assert set(report) == {ROLE_VI, ROLE_MULTI}
    assert len(factory.built) == 2
    for model in factory.built:
        assert len(model.calls) == 1
        assert model.calls[0]["n_samples"] == SAMPLE_RATE


def test_prewarm_vietnamese_only_loads_one_model(router, factory):
    router.prewarm(["vi"])
    assert len(factory.built) == 1
    assert router.loaded_roles() == (ROLE_VI,)


def test_prewarm_forces_the_right_language_per_model(router, factory):
    router.prewarm(("vi", "ko"))
    langs = {Path(m.model_dir).name: m.calls[0]["language"] for m in factory.built}
    assert langs["phowhisper-small-int8"] == "vi"
    assert langs["whisper-small-multilingual"] == "ko"


def test_prewarm_is_idempotent_and_never_reloads(router, factory):
    router.prewarm(("vi", "en"))
    router.prewarm(("vi", "en"))
    assert len(factory.built) == 2


def test_prewarm_survives_a_failing_dummy_inference(model_dirs, monkeypatch):
    vi_dir, multi_dir = model_dirs

    class Exploding(FakeModel):
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("kernel selection blew up")

    cfg = AsrRouterConfig(vi_model_dir=str(vi_dir), multi_model_dir=str(multi_dir))
    router = AsrRouter(config=cfg, loader=lambda d, c: Exploding(d, c), verbose=False)
    report = router.prewarm(["vi"])
    assert "blew up" in report[ROLE_VI]["error"]
    assert router.is_loaded(ROLE_VI)


# ---------------------------------------------------------------- footprint / attribution

def test_footprint_reports_per_model_disk_size(router):
    fp = router.footprint()
    assert fp["models"][ROLE_VI]["disk_bytes"] == 2048
    assert fp["models"][ROLE_MULTI]["disk_bytes"] == 4096
    assert fp["both_models_disk_bytes"] == 6144


def test_footprint_separates_resident_from_total(router):
    router.transcribe(audio(), "vi")
    fp = router.footprint()
    assert fp["resident_disk_bytes"] == 2048
    assert fp["both_models_disk_bytes"] == 6144
    assert fp["loaded_roles"] == [ROLE_VI]


def test_footprint_excludes_the_hf_download_cache(model_dirs, factory):
    vi_dir, multi_dir = model_dirs
    cache = multi_dir / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "blob").write_bytes(b"\0" * 100_000)
    cfg = AsrRouterConfig(vi_model_dir=str(vi_dir), multi_model_dir=str(multi_dir))
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    assert router.footprint()["models"][ROLE_MULTI]["disk_bytes"] == 4096


def test_per_call_latency_and_attribution_are_reported(router):
    result = router.transcribe(audio(seconds=2.0), "zh")
    assert isinstance(result, TranscriptionResult)
    assert result.latency_ms >= 0.0
    assert result.audio_seconds == pytest.approx(2.0)
    assert result.rtf == pytest.approx((result.latency_ms / 1000) / 2.0)
    assert result.role == ROLE_MULTI
    assert result.to_dict()["role"] == ROLE_MULTI


def test_model_info_accumulates_call_cost(router):
    router.transcribe(audio(), "en")
    router.transcribe(audio(), "ko")
    info = router.model_info(ROLE_MULTI)
    assert info.calls == 2
    assert info.total_decode_ms >= 0.0
    assert info.loaded is True
    assert router.model_info(ROLE_VI).calls == 0


# ---------------------------------------------------------------- env overrides

def test_env_overrides_model_paths_and_compute_type(monkeypatch):
    monkeypatch.setenv("VOICEAI_ASR_VI_DIR", "/tmp/vi-model")
    monkeypatch.setenv("VOICEAI_ASR_MULTI_DIR", "/tmp/multi-model")
    monkeypatch.setenv("VOICEAI_ASR_COMPUTE_TYPE", "int8_float16")
    import importlib

    reloaded = importlib.reload(asr_router)
    try:
        cfg = reloaded.AsrRouterConfig()
        assert cfg.vi_model_dir == "/tmp/vi-model"
        assert cfg.multi_model_dir == "/tmp/multi-model"
        assert cfg.compute_type == "int8_float16"
    finally:
        for key in ("VOICEAI_ASR_VI_DIR", "VOICEAI_ASR_MULTI_DIR", "VOICEAI_ASR_COMPUTE_TYPE"):
            monkeypatch.delenv(key, raising=False)
        importlib.reload(asr_router)


def test_compute_type_reaches_the_loader(model_dirs, factory):
    vi_dir, multi_dir = model_dirs
    cfg = AsrRouterConfig(
        vi_model_dir=str(vi_dir), multi_model_dir=str(multi_dir), compute_type="float16"
    )
    router = AsrRouter(config=cfg, loader=factory, verbose=False)
    router.transcribe(audio(), "vi")
    assert factory.built[0].cfg.compute_type == "float16"
    assert router.footprint()["models"][ROLE_VI]["compute_type"] == "float16"


# ---------------------------------------------------------------- lang_router composition

def test_transcribe_route_rejects_post_asr_detection_and_accepts_a_pin(router, model_dirs):
    from lang_router import DirectionRouter

    vi_dir, multi_dir = model_dirs
    direction = DirectionRouter(pair=("vi", "en"))

    vi_decision = direction.route("Xin chào, tôi muốn hỏi về giá sản phẩm này")
    assert vi_decision.source == "vi"
    with pytest.raises(ValueError, match="selected before decoding"):
        router.transcribe_route(audio(), vi_decision)

    en_decision = direction.route("Hello, could you please tell me the price of this")
    assert en_decision.source == "en"
    with pytest.raises(ValueError, match="selected before decoding"):
        router.transcribe_route(audio(), en_decision)

    direction.set_manual_source("en")
    pinned = direction.route("")
    assert router.transcribe_route(audio(), pinned).model_path == str(multi_dir)
    assert vi_dir.exists()


def test_manual_pin_wins_and_the_asr_follows_it(router, model_dirs):
    from lang_router import DirectionRouter

    vi_dir, multi_dir = model_dirs
    direction = DirectionRouter(pair=("vi", "ko"), manual_source="ko")
    decision = direction.route("Xin chào tôi là người Việt Nam")
    assert decision.manual and decision.source == "ko"
    assert router.transcribe_route(audio(), decision).model_path == str(multi_dir)
    assert vi_dir.exists()


# ---------------------------------------------------------------- audio helpers

def test_resample_leaves_16k_untouched():
    sig = np.linspace(-1, 1, 16000, dtype=np.float32)
    assert resample_to_16k(sig, 16000) is not None
    assert len(resample_to_16k(sig, 16000)) == 16000


def test_resample_changes_length_proportionally():
    sig = np.zeros(44100, dtype=np.float32)
    assert len(resample_to_16k(sig, 44100)) == 16000


def test_transcribe_resamples_before_decoding(router, factory):
    router.transcribe(np.zeros(44100, dtype=np.float32), "vi", sr=44100)
    assert factory.built[0].calls[0]["n_samples"] == 16000


def test_wav_loader_reads_a_repo_wav_if_present(router, factory):
    wav = Path(__file__).resolve().parent / "smoke_R002_en.wav"
    if not wav.exists():
        pytest.skip("no repo wav available")
    pcm = load_wav_mono16k(wav)
    assert pcm.dtype == np.float32 and pcm.ndim == 1 and len(pcm) > 0
    result = router.transcribe_file(wav, "en")
    assert result.role == ROLE_MULTI and result.audio_seconds > 0


# ---------------------------------------------------------------- real-model benchmark

@pytest.mark.skipif(
    not os.environ.get("VOICEAI_RUN_REAL_ASR"),
    reason="needs the real CT2 weights; set VOICEAI_RUN_REAL_ASR=1",
)
def test_real_benchmark_runs():
    out = asr_router.run_benchmark(repeats=1)
    assert out["footprint"]["both_models_disk_bytes"] > 0


def test_mandarin_gets_a_simplified_script_prompt():
    """Whisper drifts to Traditional on ~1/3 of Mandarin utterances; the prompt
    is what keeps the transcript in the script the rest of the pipeline expects."""
    router = AsrRouter(AsrRouterConfig(), loader=FakeFactory(), verbose=False)
    kwargs = router._decode_kwargs("zh")
    assert "简体中文" in kwargs["initial_prompt"]
    assert kwargs["language"] == "zh"


def test_other_languages_get_no_mandarin_prompt():
    router = AsrRouter(AsrRouterConfig(), loader=FakeFactory(), verbose=False)
    for lang in ("en", "ko"):
        assert "initial_prompt" not in router._decode_kwargs(lang)


def test_mandarin_prompt_can_be_disabled():
    router = AsrRouter(AsrRouterConfig(zh_initial_prompt=""), loader=FakeFactory(),
                       verbose=False)
    assert "initial_prompt" not in router._decode_kwargs("zh")
