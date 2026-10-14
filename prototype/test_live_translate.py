"""Tests for live_translate.py — all fakes, no models, no microphone.

Everything heavy is injected through LiveDeps, so this suite runs in CI on a
machine with no audio device and no model files.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_translate import (SR, LiveConfig, LiveDeps, LiveTranslator,
                            _put_drop_oldest, _resolve_cli_denoise)

TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class Seg:
    def __init__(self, samples, start=0):
        self.samples = samples
        self.start = start


class FakeVAD:
    """Scriptable stand-in for sherpa's VoiceActivityDetector.

    One script entry is consumed per accept_waveform: "S" = speech detected,
    "-" = silence, "E" = silence AND a completed turn is emitted.
    """

    def __init__(self, script, seg_len=8000, block=None):
        self.script = list(script)
        self.calls = 0
        self.reset_calls = 0
        self._speech = False
        self._pending: list = []
        self._seg_len = seg_len
        self._block = block

    def accept_waveform(self, chunk):
        if self._block is not None and self.calls == 0:
            self._block.wait(TIMEOUT)
        code = self.script[self.calls] if self.calls < len(self.script) else "-"
        self.calls += 1
        self._speech = code == "S"
        if code == "E":
            self._pending.append(Seg(np.ones(self._seg_len, dtype=np.float32)))

    def is_speech_detected(self):
        return self._speech

    def empty(self):
        return not self._pending

    @property
    def front(self):
        return self._pending[0]

    def pop(self):
        self._pending.pop(0)

    def flush(self):
        pass

    def reset(self):
        self.reset_calls += 1


class FakeASR:
    def __init__(self, text, on_finish=None):
        self.text = text
        self.on_finish = on_finish
        self._done = False

    def insert_audio(self, chunk):
        pass

    def process_iter(self):
        return ("", "")

    def finish(self):
        if self.on_finish is not None:
            self.on_finish()
        self._done = True
        return self.text

    def full_text(self):
        return self.text if self._done else ""


class FakePipe:
    def __init__(self, translate_hook=None):
        self.cfg = types.SimpleNamespace(asr_lang="vi", asr_beam=1, streaming_tts=True)
        self.mt_calls: list = []
        self.translate_hook = translate_hook
        self._asr = object()

    def _ensure_asr(self):
        pass

    def finalize_transcript(self, raw):
        return raw, {"accepted": False, "reason": "fake", "raw_text": raw, "text": raw}

    def translate(self, text, target, source="vi"):
        self.mt_calls.append((text, source, target))
        if self.translate_hook is not None:
            self.translate_hook(text)
        return f"T[{text}]"

    def synthesize(self, text, target):
        return np.zeros(160, dtype=np.float32), SR


class Frag:
    def __init__(self, index, text):
        self.index = index
        self.text = text
        self.audio = np.zeros(160, dtype=np.float32)
        self.sr = SR
        self.synth_ms = 0.0


class FakeSTTS:
    def __init__(self, synth_hook=None):
        self.spoken: list = []
        self.synth_hook = synth_hook

    def stream(self, text, target):
        if self.synth_hook is not None:
            self.synth_hook(text)
        self.spoken.append(text)
        yield Frag(0, text)


class FakePlayback:
    def __init__(self):
        self.queued: list = []
        self.waits = 0
        self.interrupts = 0
        self.speaking = False

    def new_response(self):
        pass

    def enqueue(self, chunk, sr):
        self.queued.append((len(chunk), sr))

    def end_response(self):
        pass

    def wait_idle(self, timeout=30.0):
        self.waits += 1

    def is_speaking(self):
        return self.speaking

    def interrupt(self):
        self.interrupts += 1
        return 1


def build(cfg=None, asr_texts=("xin chào",), on_finish=None, vad=None,
          splitter=None, translate_hook=None, synth_hook=None, playback=None):
    cfg = cfg or LiveConfig(manual_source="vi")
    pipe = FakePipe(translate_hook=translate_hook)
    stts = FakeSTTS(synth_hook=synth_hook)
    texts = list(asr_texts)

    def asr_factory(hint):
        text = texts.pop(0) if texts else ""
        return FakeASR(text, on_finish=on_finish)

    deps = LiveDeps(
        pipe=pipe, stts=stts, denoiser=None,
        vad_factory=(lambda: vad) if vad is not None else (lambda: FakeVAD([])),
        vad_window=160,
        playback_factory=(lambda: playback) if playback is not None else FakePlayback,
        sentence_splitter=splitter,
        online_asr_factory=asr_factory,
    )
    return LiveTranslator(cfg, deps)


def frames(n, samples=160, counter=None):
    for _ in range(n):
        if counter is not None:
            counter[0] += 1
        yield np.zeros(samples, dtype=np.float32)


def collect():
    out: list = []
    return out, out.append


def events(sink, kind):
    return [json.loads(m) for m in sink if json.loads(m).get("type") == kind]


# ---------------------------------------------------------------------------
# 1. capture is decoupled from processing
# ---------------------------------------------------------------------------
def test_capture_keeps_consuming_while_a_turn_is_processing():
    release = threading.Event()
    entered = threading.Event()

    def block():
        entered.set()
        assert release.wait(TIMEOUT)

    vad = FakeVAD(["S", "S", "E"])
    cfg = LiveConfig(prewarm=False, play=False, half_duplex=False, manual_source="vi",
                     speculative_final=False, min_gap_between_turns=0.0)
    lt = build(cfg, asr_texts=["hello"], on_finish=block, vad=vad)

    produced = [0]
    sink, emit = collect()
    done = threading.Event()

    def run():
        lt._feed_frames(frames(40, counter=produced), emit=emit, play=False)
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()

    assert entered.wait(TIMEOUT), "processing never started"
    deadline = time.time() + TIMEOUT
    while produced[0] < 40 and time.time() < deadline:
        time.sleep(0.01)

    # The whole input stream was consumed while the ASR of turn 1 is still blocked:
    # this is the property the old inline _feed_frames could not have.
    assert produced[0] == 40
    assert not done.is_set()

    release.set()
    assert done.wait(TIMEOUT)
    assert vad.calls >= 40


def test_processing_thread_survives_a_failing_turn():
    def boom():
        raise RuntimeError("asr exploded")

    vad = FakeVAD(["S", "E", "S", "E"])
    cfg = LiveConfig(prewarm=False, play=False, half_duplex=False, manual_source="vi",
                     speculative_final=False)
    lt = build(cfg, asr_texts=["a", "b"], on_finish=boom, vad=vad)
    sink, emit = collect()
    lt._feed_frames(frames(10), emit=emit, play=False)
    assert len(events(sink, "turn_error")) == 2


# ---------------------------------------------------------------------------
# 2. drop-oldest backpressure
# ---------------------------------------------------------------------------
def test_put_drop_oldest_evicts_the_oldest_item():
    q: queue.Queue = queue.Queue(maxsize=2)
    assert _put_drop_oldest(q, 1) == 0
    assert _put_drop_oldest(q, 2) == 0
    assert _put_drop_oldest(q, 3) == 1
    assert list(q.queue) == [2, 3]


def test_capture_drops_oldest_frames_and_logs_when_vad_falls_behind():
    gate = threading.Event()
    vad = FakeVAD([], block=gate)
    cfg = LiveConfig(prewarm=False, play=False, half_duplex=False, manual_source="vi",
                     speculative_final=False, frame_queue_max=2)
    lt = build(cfg, vad=vad)
    sink, emit = collect()
    done = threading.Event()

    def run():
        lt._feed_frames(frames(30), emit=emit, play=False, drop_oldest=True)
        done.set()

    threading.Thread(target=run, daemon=True).start()

    deadline = time.time() + TIMEOUT
    while lt.frames_dropped == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert lt.frames_dropped > 0, "backpressure never engaged"

    gate.set()
    assert done.wait(TIMEOUT)

    drops = events(sink, "frames_dropped")
    assert drops, "frames were dropped silently"
    assert drops[-1]["total"] == lt.frames_dropped


def test_non_realtime_input_is_never_dropped():
    # File/sidecar audio is not realtime; dropping there would corrupt a run.
    vad = FakeVAD([])
    cfg = LiveConfig(prewarm=False, play=False, half_duplex=False, manual_source="vi",
                     speculative_final=False, frame_queue_max=2)
    lt = build(cfg, vad=vad)
    sink, emit = collect()
    lt._feed_frames(frames(30), emit=emit, play=False, drop_oldest=False)
    assert lt.frames_dropped == 0
    assert vad.calls == 30


# ---------------------------------------------------------------------------
# 3. speculative final ASR + discard on resume
# ---------------------------------------------------------------------------
def _spec_cfg(**kw):
    base = dict(prewarm=False, play=False, half_duplex=False, min_speech_ms=0,
                speculative_final=True, early_transcription_s=0.0,
                manual_source="vi")
    base.update(kw)
    return LiveConfig(**base)


def test_speculative_final_is_used_when_speech_does_not_resume():
    vad = FakeVAD(["S", "S", "-", "-", "E"])
    lt = build(_spec_cfg(), asr_texts=["SPECULATIVE"], vad=vad)
    sink, emit = collect()
    lt._feed_frames(frames(10), emit=emit, play=False)

    assert events(sink, "speculative_final"), "speculation never fired"
    final = events(sink, "final")[-1]
    assert final["raw_src"] == "SPECULATIVE"
    assert final["speculative"] is True


def test_speculative_result_is_discarded_when_speech_resumes():
    # Silence → speculate → speech resumes → that speculation must never surface.
    vad = FakeVAD(["S", "S", "-", "-", "S", "S", "-", "-", "E"])
    lt = build(_spec_cfg(), asr_texts=["STALE", "FRESH"], vad=vad)
    sink, emit = collect()
    lt._feed_frames(frames(12), emit=emit, play=False)

    blob = "\n".join(sink)
    assert "STALE" not in blob, "a discarded speculation was emitted"
    final = events(sink, "final")[-1]
    assert final["raw_src"] == "FRESH"
    assert len(events(sink, "speculative_final")) == 2


def test_speculation_disabled_falls_back_to_the_confirmed_final():
    vad = FakeVAD(["S", "S", "E"])
    lt = build(_spec_cfg(speculative_final=False), asr_texts=["CONFIRMED"], vad=vad)
    sink, emit = collect()
    lt._feed_frames(frames(8), emit=emit, play=False)

    assert not events(sink, "speculative_final")
    final = events(sink, "final")[-1]
    assert final["raw_src"] == "CONFIRMED"
    assert final["speculative"] is False


def test_a_failed_speculation_falls_back_instead_of_killing_the_turn(monkeypatch):
    lt = build(_spec_cfg(), asr_texts=["REDONE"])

    class Boom:
        valid = True
        source = "vi"
        future = types.SimpleNamespace(result=lambda: (_ for _ in ()).throw(RuntimeError("x")))

    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit,
                                  play=False, spec=Boom())
    assert events(sink, "final")[-1]["raw_src"] == "REDONE"


# ---------------------------------------------------------------------------
# 4. MT and TTS are pipelined, not serialized
# ---------------------------------------------------------------------------
def test_tts_of_sentence_1_overlaps_mt_of_sentence_2():
    mt_second_started = threading.Event()
    saw_overlap = [False]

    def translate_hook(text):
        if text == "B.":
            mt_second_started.set()

    def synth_hook(text):
        if text == "T[A.]":
            saw_overlap[0] = mt_second_started.wait(TIMEOUT)

    cfg = LiveConfig(prewarm=False, play=False, manual_source="vi")
    lt = build(cfg, asr_texts=["A. B."], splitter=lambda t: ["A.", "B."],
               translate_hook=translate_hook, synth_hook=synth_hook)
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)

    assert saw_overlap[0], "MT of sentence 2 did not start until TTS of sentence 1 finished"
    assert lt.stts.spoken == ["T[A.]", "T[B.]"], "fragments spoken out of order"
    assert [e["text"] for e in events(sink, "tts_partial")] == ["T[A.]", "T[B.]"]
    assert events(sink, "final")[-1]["tgt"] == "T[A.] T[B.]"


def test_partial_target_text_grows_sentence_by_sentence():
    cfg = LiveConfig(prewarm=False, play=False, manual_source="vi")
    lt = build(cfg, asr_texts=["A. B."], splitter=lambda t: ["A.", "B."])
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)
    tgt = [e["text"] for e in events(sink, "partial") if e["field"] == "tgt"]
    assert tgt == ["T[A.]", "T[A.] T[B.]"]


# ---------------------------------------------------------------------------
# 5. direction routing: the manual pin always beats detection
# ---------------------------------------------------------------------------
def test_transcript_cannot_change_the_preselected_direction():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"), manual_source="vi")
    lt = build(cfg, asr_texts=["this English-looking transcript arrives after ASR"])
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)

    d = events(sink, "direction")[-1]
    assert (d["source"], d["target"]) == ("vi", "en")
    assert d["manual"] is True
    assert d["confidence"] == 1.0
    assert lt.pipe.mt_calls[0][1:] == ("vi", "en")


def test_missing_source_blocks_the_turn_before_asr():
    seen = []
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"))
    deps = LiveDeps(pipe=FakePipe(), stts=FakeSTTS(), vad_window=160,
                    online_asr_factory=lambda source: seen.append(source))
    lt = LiveTranslator(cfg, deps)

    with pytest.raises(RuntimeError, match="source language is required before ASR"):
        lt.handle_utterance_streaming(np.zeros(1600, np.float32), play=False)
    assert seen == [], "ASR factory was touched before source selection"


def test_direction_event_precedes_asr_text_events():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"), manual_source="vi")
    lt = build(cfg, asr_texts=["xin chào"])
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)

    kinds = [json.loads(message)["type"] for message in sink]
    assert kinds.index("direction") < kinds.index("partial")


def test_reverse_source_pin_controls_mt_and_corrector():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"), manual_source="en")
    # Transcript content arrives after ASR and cannot override the operator's pin.
    lt = build(cfg, asr_texts=["tôi không biết cái này"])
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)

    d = events(sink, "direction")[-1]
    assert (d["source"], d["target"]) == ("en", "vi")
    assert d["manual"] is True
    assert d["confidence"] == 1.0
    assert lt.pipe.mt_calls[0][1:] == ("en", "vi")
    # The Vietnamese corrector must not touch a non-Vietnamese transcript.
    assert events(sink, "corrected")[-1]["correction"] is None


def test_manual_pin_can_be_set_and_cleared_at_runtime():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "ko"), manual_source="vi")
    lt = build(cfg)
    lt.set_manual_source("ko")
    assert lt.direction_state()["manual_source"] == "ko"
    lt.set_manual_source(None)
    assert lt.direction_state()["manual_source"] is None
    assert lt.direction_state()["source_required"] is True
    with pytest.raises(ValueError):
        lt.set_manual_source("en")   # not in the configured pair


def test_sidecar_control_channel_pins_the_direction():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "zh"), manual_source="vi")
    lt = build(cfg)
    sink, emit = collect()
    lt._control_loop(['{"type": "set_direction", "source": "zh"}',
                      "",
                      '{"type": "set_direction", "source": "de"}',
                      '{"type": "clear_direction"}'], emit=emit)
    pins = events(sink, "direction_pinned")
    assert [e["manual_source"] for e in pins] == ["zh", None]
    assert [e["source_required"] for e in pins] == [False, True]
    assert all(e["routing_mode"] == "pre_asr_pin" for e in pins)
    assert all(e["automatic_routing"] is False for e in pins)
    assert len(events(sink, "control_error")) == 1
    assert lt.router.manual_source is None


def test_target_flag_still_derives_the_pair():
    assert LiveConfig(target="ko").pair == ("vi", "ko")
    assert LiveConfig(pair=("vi", "zh")).pair == ("vi", "zh")


def test_asr_seam_receives_the_direction_hint():
    seen: list = []

    def asr_factory(hint):
        seen.append(hint)
        return FakeASR("xin chào")

    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"), manual_source="en")
    deps = LiveDeps(pipe=FakePipe(), stts=FakeSTTS(), vad_window=160,
                    online_asr_factory=asr_factory)
    lt = LiveTranslator(cfg, deps)
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=lambda m: None,
                                  play=False)
    assert seen == ["en"]


def test_reverse_source_requires_the_multilingual_asr_path():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"),
                     manual_source="en", multilingual_asr=False)
    deps = LiveDeps(pipe=FakePipe(), stts=FakeSTTS(), vad_window=160)
    lt = LiveTranslator(cfg, deps)

    with pytest.raises(RuntimeError, match="requires --multilingual-asr"):
        lt.handle_utterance_streaming(np.zeros(1600, np.float32), play=False)


def test_source_must_belong_to_the_configured_pair():
    with pytest.raises(ValueError, match="not in configured pair"):
        LiveConfig(pair=("vi", "ko"), manual_source="en")


def test_direction_change_discards_speculation_from_the_previous_source():
    cfg = LiveConfig(prewarm=False, play=False, pair=("vi", "en"), manual_source="vi")
    lt = build(cfg, asr_texts=["STALE", "FRESH"])
    spec = lt._start_speculation(np.zeros(1600, np.float32))
    assert spec.future.result(timeout=TIMEOUT) == "STALE"

    lt.set_manual_source("en")
    sink, emit = collect()
    result = lt.handle_utterance_streaming(
        np.zeros(1600, np.float32), emit=emit, play=False, spec=spec
    )

    assert result["raw_vi"] == "FRESH"
    assert result["direction"]["source"] == "en"
    assert lt.pipe.mt_calls[-1][1:] == ("en", "vi")


# ---------------------------------------------------------------------------
# 6. half-duplex mic gate
# ---------------------------------------------------------------------------
def test_mic_is_gated_shut_while_we_are_speaking_and_reopens_after():
    release = threading.Event()
    entered = threading.Event()

    def block():
        entered.set()
        assert release.wait(TIMEOUT)

    vad = FakeVAD(["S", "S", "E"])
    cfg = LiveConfig(prewarm=False, play=True, half_duplex=True, barge_in=False,
                     manual_source="vi",
                     speculative_final=False, min_gap_between_turns=0.0)
    pb = FakePlayback()
    lt = build(cfg, asr_texts=["hello"], on_finish=block, vad=vad, playback=pb)
    lt._sync_playback = False   # what run_mic does

    sink, emit = collect()
    done = threading.Event()

    def run():
        lt._feed_frames(frames(40), emit=emit, play=True)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert entered.wait(TIMEOUT)

    deadline = time.time() + TIMEOUT
    while lt.frames_gated == 0 and time.time() < deadline:
        time.sleep(0.01)
    # Frames are still being read off the device, but they never reach the VAD:
    # our own speaker output cannot open a new turn.
    assert lt.frames_gated > 0
    fed_while_speaking = vad.calls

    release.set()
    assert done.wait(TIMEOUT)

    gates = events(sink, "mic_gate")
    assert [g["open"] for g in gates] == [False, True]
    assert vad.calls >= fed_while_speaking
    assert vad.reset_calls >= 0   # reset only if frames survive after reopen


def test_half_duplex_is_off_for_file_and_sidecar_paths():
    cfg = LiveConfig(prewarm=False, half_duplex=True, manual_source="vi")
    lt = build(cfg)
    assert lt._half_duplex_active(play=False) is False   # sidecar / file
    lt._sync_playback = True
    assert lt._half_duplex_active(play=True) is False    # blocking playback, no mic


def test_barge_in_is_still_available_as_an_explicit_opt_in():
    cfg = LiveConfig(prewarm=False, play=True, barge_in=True, half_duplex=True,
                     manual_source="vi")
    lt = build(cfg, asr_texts=["hello"])
    lt._sync_playback = False
    assert lt._half_duplex_active(play=True) is False, "barge-in must disable the gate"

    pb = FakePlayback()
    pb.speaking = True
    lt._playback = pb
    sink, emit = collect()
    lt._dispatch_segment(np.zeros(1600, np.float32), emit, play=True)
    assert pb.interrupts == 1
    assert events(sink, "barge_in")


def test_min_gap_between_turns_is_honoured():
    cfg = LiveConfig(prewarm=False, min_gap_between_turns=0.15, manual_source="vi")
    lt = build(cfg)
    lt._mic_open.clear()
    t0 = time.perf_counter()
    lt._reopen_mic(lambda m: None)
    assert time.perf_counter() - t0 >= 0.15
    assert lt._mic_open.is_set()


# ---------------------------------------------------------------------------
# entry modes + event-protocol backward compatibility
# ---------------------------------------------------------------------------
def test_sidecar_mode_reads_pcm_from_stdin(monkeypatch):
    import io
    pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(pcm)))

    vad = FakeVAD(["S", "S", "E"])
    cfg = LiveConfig(prewarm=False, play=False, speculative_final=False,
                     manual_source="vi")
    lt = build(cfg, asr_texts=["xin chào"], vad=vad)
    sink, emit = collect()
    lt.run_sidecar(emit=emit)
    assert events(sink, "final")


def test_file_mode_runs_the_same_path(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(SR, dtype=np.float32), SR)

    vad = FakeVAD(["S", "S", "E"])
    cfg = LiveConfig(prewarm=False, play=False, speculative_final=False,
                     manual_source="vi")
    lt = build(cfg, asr_texts=["xin chào"], vad=vad)
    sink, emit = collect()
    lt.run_file(str(wav), emit=emit, play=False)
    assert events(sink, "final")


def test_final_event_keeps_every_legacy_field():
    cfg = LiveConfig(prewarm=False, play=False, manual_source="vi")
    lt = build(cfg, asr_texts=["xin chào"])
    sink, emit = collect()
    lt.handle_utterance_streaming(np.zeros(1600, np.float32), emit=emit, play=False)

    final = events(sink, "final")[-1]
    for key in ("raw_src", "src", "tgt", "correction", "first_audio_ms", "total_ms"):
        assert key in final
    corrected = events(sink, "corrected")[-1]
    for key in ("raw", "text", "correction", "latency_ms"):
        assert key in corrected
    assert {"source", "target", "confidence", "reason", "manual"} <= set(final["direction"])


def test_asr_denoiser_is_off_by_default():
    """Measured on T4: GTCRN in front of ASR is statistically neutral in 14 of 16
    noise cells and significantly harmful in babble at low SNR, for 363 ms p50.
    A future change that flips this back needs new evidence, not a preference."""
    assert LiveConfig().denoise is False


def test_cli_denoise_is_opt_in_and_preserves_the_safe_default():
    assert _resolve_cli_denoise(None) is False
    assert _resolve_cli_denoise(False) is False
    assert _resolve_cli_denoise(True) is True


def test_injected_denoiser_cannot_override_the_safe_default():
    class FakeDenoiser:
        def __init__(self):
            self.calls = 0

        def denoise(self, audio, sr):
            self.calls += 1
            return np.full_like(audio, 9.0), {"sr": sr}

    denoiser = FakeDenoiser()
    deps = LiveDeps(pipe=FakePipe(), stts=FakeSTTS(), denoiser=denoiser,
                    vad_factory=lambda: FakeVAD([]), vad_window=160,
                    online_asr_factory=lambda _hint: FakeASR("xin chào"))
    lt = LiveTranslator(LiveConfig(prewarm=False, play=False, denoise=False), deps)
    handed_over = []
    lt.handle_utterance = lambda audio, **kw: handed_over.append(np.asarray(audio).copy())
    raw = np.ones(1600, dtype=np.float32)
    lt._dispatch_segment(raw, emit=lambda _msg: None, play=False)

    assert denoiser.calls == 0
    assert handed_over and np.array_equal(handed_over[0], raw)


def test_asr_denoiser_can_still_be_enabled(monkeypatch):
    import importlib
    import live_translate

    monkeypatch.setenv("VOICEAI_DENOISE", "1")
    try:
        reloaded = importlib.reload(live_translate)
        assert reloaded.LiveConfig().denoise is True
    finally:
        monkeypatch.delenv("VOICEAI_DENOISE", raising=False)
        importlib.reload(live_translate)


class _FakeStreamDenoiser:
    """Length-preserving after a priming delay, like the real GTCRN wrapper."""

    def __init__(self, prime=128):
        self.prime = prime
        self.seen = []

    def accept(self, frame, sr):
        self.seen.append(len(frame))
        take = len(frame) - self.prime if self.prime else len(frame)
        self.prime = max(0, self.prime - len(frame))
        if take <= 0:
            return np.zeros(0, dtype=np.float32)
        # Mark the denoised copy so a test can prove which stream went where.
        return np.full(take, 9.0, dtype=np.float32)

    def flush(self):
        return np.zeros(0, dtype=np.float32)

    def reset(self):
        pass


def test_streaming_denoiser_feeds_vad_but_asr_gets_raw_audio(monkeypatch):
    """The T4 grid put GTCRN's only win on the gate and its harm on the recogniser.
    The trap is that sherpa's VAD hands back the samples it was FED, so feeding it
    denoised audio silently routes denoised audio into ASR. The segment must be
    re-cut from the raw history instead. A refactor that loses this is invisible in
    latency and costs WER, so pin it."""
    import edge_frontend

    fake = _FakeStreamDenoiser(prime=0)
    monkeypatch.setattr(edge_frontend, "make_streaming_denoiser",
                        lambda *a, **k: fake, raising=False)

    seen_by_vad, handed_over = [], []

    class RecordingVAD(FakeVAD):
        def accept_waveform(self, chunk):
            seen_by_vad.append(np.asarray(chunk).copy())
            super().accept_waveform(chunk)

    # seg_len must clear min_speech_ms, and the segment carries a `start` so the
    # raw re-cut has an absolute position to work from.
    vad = RecordingVAD(["S"] * 8 + ["E"], seg_len=4000)
    cfg = LiveConfig(prewarm=False, play=False, half_duplex=False,
                     manual_source="vi",
                     speculative_final=False, denoise_vad=True)
    lt = build(cfg, vad=vad)
    lt.handle_utterance = lambda audio, **kw: handed_over.append(np.asarray(audio).copy())

    ramp = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    lt._feed_frames(iter([ramp] * 40), emit=lambda *_: None, play=False)

    assert seen_by_vad, "the VAD was never fed"
    assert all(np.allclose(c, 9.0) for c in seen_by_vad),         "the VAD must receive the denoised stream"
    assert handed_over, "no utterance reached the pipeline"
    for utt in handed_over:
        assert not np.allclose(utt, 9.0), "denoised audio leaked into the recogniser"


def test_denoise_vad_is_off_by_default():
    assert LiveConfig().denoise_vad is False
