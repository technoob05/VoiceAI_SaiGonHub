from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
from transcript_corrector import CorrectorConfig, TranscriptCorrector


def test_submission_defaults_fail_closed_for_unproven_runtime_features():
    cfg = FastPipelineConfig()
    assert cfg.cache_enabled is False
    assert cfg.cache_two_stream is True
    assert cfg.tm_enabled is False
    assert cfg.denoise_gtcrn is False
    assert cfg.tts_auto_download is False


def _pipeline(proposal: str) -> FastCascadeS2ST:
    cfg = FastPipelineConfig(
        cache_enabled=False,
        tm_enabled=False,
        tone_vad_enabled=False,
        corrector_enabled=True,
        punct=False,
    )
    pipeline = FastCascadeS2ST(cfg)
    pipeline._corrector = TranscriptCorrector(
        CorrectorConfig(max_edit_ratio=0.35), generator=lambda _text: proposal
    )
    return pipeline


def test_final_correction_is_selected_before_translation():
    pipeline = _pipeline("Tôi đang làm việc ở nhà.")
    selected, metadata = pipeline.finalize_transcript("Tôi đang làm viêc ở nhà.")
    assert selected == "Tôi đang làm việc ở nhà."
    assert metadata["accepted"] is True


def test_unsafe_correction_falls_back_to_raw_final():
    pipeline = _pipeline("Ngày mai công ty đóng cửa.")
    raw = "Hôm nay công ty mở cửa."
    selected, metadata = pipeline.finalize_transcript(raw)
    assert selected == raw
    assert metadata["accepted"] is False
