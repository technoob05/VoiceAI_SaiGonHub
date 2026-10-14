"""live_translate.py — the always-warm streaming S2ST app.

This is the product surface that ties the fused upgrades together:

  mic → [GTCRN denoise] → Silero-VAD turn segmentation → pipeline_fast
        (LocalAgreement ASR → NLLB MT) → streaming TTS → speaker

Turn segmentation uses sherpa-onnx's Silero VoiceActivityDetector (the same model
as edge_frontend, integrated with min-silence endpointing) instead of hand-rolling
the RealtimeSTT state machine — it is on-device-ready and closes a turn the instant
the speaker pauses, which is what drives the <1.5 s turn latency.

Three run modes:
  * mic    — live microphone loop (needs sounddevice + a mic).
  * file   — feed a wav through the exact same turn→translate→speak path (no mic,
             for testing/CI and demos on machines without audio input).
  * sidecar — read raw s16le/16k/mono PCM from stdin, emit one JSON line per event
             on stdout (the my-translator IPC pattern). This is the seam a Tauri /
             on-device host process plugs into: models load once, audio streams in,
             results stream out.

All heavy models load once at startup and stay warm.

Concurrency (Round 5 — the reason this file has threads at all):
  capture thread → frame queue → VAD/segmentation thread (the caller's) → turn
  queue → processing thread (ASR→MT→TTS→playback), plus a single-slot speculative
  ASR worker. Capture must never wait on processing: the microphone is a realtime
  device, and every millisecond the read loop is blocked is audio that is *lost*,
  not delayed. Mirrors RealtimeSTT's audio_input_worker → audio_queue → recording
  worker split, with the same drop-oldest overflow policy so latency stays bounded.

Direction is selected before audio reaches ASR. The operator, host UI, or physical
push-to-talk button pins the source language; transcript language detection is not
an acoustic router and is never used to choose an ASR model.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lang_router import DirectionRouter, RouteDecision, normalize_pair  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

SR = 16_000

_EOS = object()   # queue sentinel: upstream stage finished


@dataclass
class LiveConfig:
    target: str = "en"
    # GTCRN in front of ASR is OFF by default, and that is a measured decision, not
    # caution. On a Kaggle T4, 60 VIVOS utterances x 4 synthetic noise types x 4 SNRs
    # (Results/voiceai_noise-t4-v1.json): the denoiser raised SI-SDR in all 16 cells
    # (+3.6 to +21.9 dB) yet its effect on WER was statistically indistinguishable
    # from zero in 14 of them, and significantly HARMFUL in the two babble cells --
    # catastrophically at 0 dB, where WER doubled (0.276 -> 0.560, 95% CI
    # [+0.172, +0.448]). Multi-talker babble at low SNR is precisely the factory
    # floor. It also costs 363 ms p50 / 539 ms p95, a quarter of the 1.5 s budget.
    #
    # Its one demonstrated benefit is on the VAD, not the recognizer: it eliminated
    # Silero false triggers on impulsive metallic noise (8-12.5% -> 0% across all
    # four clank cells), and each false trigger wastes a whole ASR pass. Capturing
    # that needs a STREAMING denoiser on the capture path; edge_frontend's GTCRN is
    # whole-signal, so the live loop cannot use it that way today. See
    # sherpa-onnx OnlineSpeechDenoiser -- that is the follow-up.
    denoise: bool = bool(int(os.environ.get("VOICEAI_DENOISE", "0")))
    # The other half of that measurement: GTCRN removed Silero's false triggers on
    # impulsive metallic noise entirely (8.3-12.5% -> 0% across four SNRs), and each
    # false trigger costs a whole wasted ASR pass. That win is on the GATE, so the
    # denoised stream drives the VAD while the recogniser keeps the raw audio.
    # Costs ~29 ms per 100 ms of capture (RTF 0.29) on this dev CPU.
    #
    # UPDATE 2026-08-20: that VAD benefit was a SYNTHETIC-noise artifact and did not
    # replicate on real recordings. On DEMAND PSTATION (a production hall) Silero
    # never false-triggers on the raw audio, and after GTCRN it fires on 29.2 % of
    # segments at 0 dB and 16.7 % at 5 dB - the denoiser MANUFACTURES triggers there.
    # Only DKITCHEN improved. So this flag currently has no measured benefit at all;
    # the machinery and its tests stay because the two-stream split is correct and
    # the "ASR must see raw audio" invariant is worth keeping pinned.
    # See Results/noise_robustness_README.md.
    denoise_vad: bool = bool(int(os.environ.get("VOICEAI_DENOISE_VAD", "0")))
    min_silence: float = 0.5      # end a turn after this much trailing silence
    vad_threshold: float = 0.5
    play: bool = True             # play synthesized audio (mic mode)
    streaming_asr: bool = True    # use LocalAgreement-2 streaming ASR within a turn
    asr_chunk: float = 1.5        # streaming-ASR sub-chunk size (s) within a turn
    # --- turn-taking upgrades absorbed from KoljaB/RealtimeSTT (MIT) ---
    preroll_ms: int = 300         # pre-roll spliced before each turn (recover clipped onsets)
    early_finalize: bool = True   # gate the short-turn guard (see _feed_frames)
    min_speech_ms: int = 200      # drop VAD turns whose speech span is shorter than this
    corrector: bool = True         # final transcript only; partials never enter MT
    corrector_model: str = "bmd1905/vietnamese-correction-v2"
    corrector_local_only: bool = True  # runtime is offline; prepare model once beforehand
    prewarm: bool = True            # pay cold-start before accepting microphone audio
    # --- capture/processing decoupling ---
    frame_queue_max: int = 50     # ~5 s of 100 ms frames before drop-oldest kicks in
    turn_queue_max: int = 4       # completed turns awaiting ASR→MT→TTS
    # --- speculative final ASR during the endpoint countdown ---
    speculative_final: bool = True
    early_transcription_s: float = 0.25   # fire this far into the min_silence countdown
    # --- half-duplex (own-speaker feedback guard) ---
    half_duplex: bool = True      # close the mic while we are speaking (RTranslator)
    barge_in: bool = False        # opt-in: keep the mic live so a new turn can interrupt
    min_gap_between_turns: float = 0.3    # RealtimeSTT min_gap_between_recordings
    # --- direction ---
    # `pair` is the configured two-way pair; None derives it from `target` so every
    # existing `--target xx` invocation keeps working unchanged.
    pair: Optional[tuple] = None
    manual_source: Optional[str] = None   # must be pinned before a turn reaches ASR
    # Route non-Vietnamese turns to a multilingual Whisper instead of the
    # Vietnamese specialist. Off by default: it adds a second resident ASR model
    # (~486 MB on disk) and its Mandarin/Korean accuracy is unmeasured, so it is
    # a deliberate opt-in rather than something a demo picks up silently.
    multilingual_asr: bool = False

    def __post_init__(self):
        if self.pair is None:
            other = "en" if self.target == "vi" else self.target
            self.pair = ("vi", other)
        self.pair = normalize_pair(self.pair)
        if self.manual_source is not None:
            source = self.manual_source.strip().lower()
            if source not in self.pair:
                raise ValueError(
                    f"source {self.manual_source!r} is not in configured pair {self.pair}"
                )
            self.manual_source = source


@dataclass
class LiveDeps:
    """Injection seam for every heavy component.

    Production leaves this empty and `LiveTranslator.__init__` loads the real
    models; tests pass fakes and never touch a model file. Nothing in this class
    is optional-for-convenience — each field is a thing that would otherwise
    force a multi-gigabyte download into a unit test.
    """

    pipe: object | None = None            # FastCascadeS2ST-compatible
    stts: object | None = None            # StreamingTTS-compatible (.stream(text, lang))
    denoiser: object | None = None        # .denoise(audio, sr) -> (audio, meta)
    vad_factory: Optional[Callable[[], object]] = None   # () -> sherpa VAD-compatible
    vad_window: Optional[int] = None      # samples per VAD accept_waveform call
    playback_factory: Optional[Callable[[], object]] = None
    sentence_splitter: Optional[Callable[[str], list]] = None
    # ---- ASR SEAM (read before wiring the bidirectional ASR module) ----
    # Called as online_asr_factory(source) -> an OnlineASR-compatible object
    # (insert_audio / process_iter / finish / full_text). `source` is the language
    # explicitly selected before this turn reaches ASR; it is never None.
    #
    # Leave this None and the behaviour depends on LiveConfig.multilingual_asr:
    # off (default) always returns the Vietnamese-specialised PhoWhisper stream;
    # on routes the hint through asr_router.AsrRouter, which owns a second
    # multilingual model for en/zh/ko. Override this field only to inject a fake
    # in tests or to plug in a different ASR entirely.
    online_asr_factory: Optional[Callable[[Optional[str]], object]] = None


class _Speculation:
    """One in-flight speculative final ASR, tied to the turn that spawned it.

    `valid` is cleared the moment speech resumes during the countdown. It is set
    only by the VAD thread and read only after that thread has handed the turn
    over, so a discarded speculation can never reach an emit().
    """

    __slots__ = ("future", "valid", "source")

    def __init__(self, future, source: str):
        self.future = future
        self.valid = True
        self.source = source

    def discard(self) -> None:
        self.valid = False
        self.future.cancel()


def _put_drop_oldest(q: "queue.Queue", item) -> int:
    """Bounded put that evicts the OLDEST item on overflow, returning how many
    were dropped. Dropping the newest frame would keep a stale backlog and grow
    perceived latency without bound (RealtimeSTT recording.py handle_buffer_overflow)."""
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                pass


class LiveTranslator:
    def __init__(self, cfg: LiveConfig | None = None, deps: LiveDeps | None = None):
        self.cfg = cfg or LiveConfig()
        self.deps = deps or LiveDeps()
        self.router = DirectionRouter(self.cfg.pair, manual_source=self.cfg.manual_source)
        # One lock over ALL ASR use: the speculative worker and the turn worker
        # share a single faster-whisper model, which is not safe to drive from two
        # threads at once.
        self._asr_lock = threading.Lock()
        self._spec_executor: Optional[ThreadPoolExecutor] = None
        self._mic_open = threading.Event()
        self._mic_open.set()
        self.frames_dropped = 0     # capture-side overflow (observable in tests/logs)
        self.frames_gated = 0       # frames discarded because our own speaker was live
        self._OnlineASR = None
        self._asr_router = None
        self._playback = None
        # File/sidecar drain playback synchronously; the mic loop plays in the
        # background so the next turn can barge-in (set False in run_mic).
        self._sync_playback = True

        if self.deps.pipe is not None:
            self._attach_injected()
        else:
            self._load_runtime()

    # ---- construction ----
    def _attach_injected(self):
        self.pipe = self.deps.pipe
        self.stts = self.deps.stts
        # Injection must not silently override the measured-safe default. The
        # device supervisor may keep an optional frontend resident, but live
        # audio only uses it after an explicit runtime opt-in.
        self._denoiser = (self.deps.denoiser
                          if self.cfg.denoise or self.cfg.denoise_vad else None)
        self._window = self.deps.vad_window or 512
        self._new_vad = self.deps.vad_factory or (lambda: None)
        if self.deps.sentence_splitter is not None:
            self._split_sentences = self.deps.sentence_splitter
        else:
            from streaming_tts import split_sentences
            self._split_sentences = split_sentences

    def _load_runtime(self):
        if (self.cfg.manual_source not in (None, "vi")
                and (not self.cfg.streaming_asr or not self.cfg.multilingual_asr)):
            raise ValueError(
                f"source {self.cfg.manual_source!r} requires streaming ASR with "
                "--multilingual-asr"
            )
        # Pipeline with all edge upgrades on (streaming TTS + Piper backend).
        import os
        os.environ.setdefault("VOICEAI_TTS_BACKEND", "piper")
        os.environ.setdefault("VOICEAI_STREAMING_TTS", "1")
        from pipeline_fast import FastCascadeS2ST, FastPipelineConfig, _split_sentences
        pipe_cfg = FastPipelineConfig()
        pipe_cfg.corrector_enabled = self.cfg.corrector
        pipe_cfg.corrector_model = self.cfg.corrector_model
        pipe_cfg.corrector_local_only = self.cfg.corrector_local_only
        self.pipe = FastCascadeS2ST(pipe_cfg)
        self._split_sentences = _split_sentences
        from streaming_tts import StreamingTTS
        self.stts = StreamingTTS(lambda tx, tg: self.pipe.synthesize(tx, tg))
        self._denoiser = None
        if self.cfg.denoise:
            from edge_frontend import get_denoiser
            self._denoiser = get_denoiser()
        # sherpa Silero VAD detector for turn segmentation.
        import sherpa_onnx
        from edge_frontend import SILERO_URL, _ensure
        vc = sherpa_onnx.VadModelConfig()
        vc.silero_vad.model = _ensure(SILERO_URL)
        vc.silero_vad.min_silence_duration = self.cfg.min_silence
        vc.silero_vad.threshold = self.cfg.vad_threshold
        vc.sample_rate = SR
        self._window = vc.silero_vad.window_size
        self._new_vad = lambda: sherpa_onnx.VoiceActivityDetector(
            vc, buffer_size_in_seconds=100)
        # Streaming-ASR: reuse the pipeline's loaded faster-whisper model (LocalAgreement-2
        # incremental decoding). Warm it now so the first turn does not pay cold-load.
        if self.cfg.streaming_asr:
            from streaming_asr import OnlineASR
            self._OnlineASR = OnlineASR
            if self.cfg.multilingual_asr:
                from asr_router import AsrRouter
                self._asr_router = AsrRouter()
                if self.cfg.prewarm:
                    self._asr_router.prewarm_pair(self.cfg.pair)
            else:
                self.pipe._ensure_asr()
        if self.cfg.corrector:
            corrector = self.pipe._ensure_corrector()
            if not corrector.prewarm():
                print(f"[warn] corrector unavailable; raw ASR fallback active: "
                      f"{corrector._load_error}", file=sys.stderr)
            else:
                corrector.correct("xin chào")
        if self.cfg.prewarm:
            # The process is a persistent sidecar: initialize every downstream
            # engine before listening so cold-start is not charged to turn latency.
            self.pipe._ensure_mt()
            src, tgt = self.cfg.pair
            self.pipe.translate("xin chào", tgt, source=src)
            self.pipe.synthesize("Ready.", tgt)
            print(f"[ready] all local models warm; pair={self.cfg.pair[0]}<->{self.cfg.pair[1]}",
                  file=sys.stderr)

    def _get_playback(self):
        if self._playback is None:
            if self.deps.playback_factory is not None:
                self._playback = self.deps.playback_factory()
            else:
                from playback import Playback
                self._playback = Playback()
        return self._playback

    # ---- direction control (CLI, sidecar control channel, physical PTT buttons) ----
    def set_manual_source(self, lang: Optional[str]) -> None:
        """Pin the source language before ASR. None pauses future ASR turns."""
        self.router.set_manual_source(lang)

    def direction_state(self) -> dict:
        return {"pair": list(self.router.pair),
                "manual_source": self.router.manual_source,
                "last_source": self.router.last_source,
                "source_required": self.router.manual_source is None,
                "routing_mode": "pre_asr_pin",
                "automatic_routing": False}

    def _source_before_asr(self) -> str:
        source = self.router.manual_source
        if source is None:
            raise RuntimeError(
                "source language is required before ASR; set --source/--manual-source "
                "or send a set_direction control message"
            )
        return source

    def _decision_before_asr(self, source: str) -> RouteDecision:
        return self.router.route("", manual_source=source)

    def _direction_event(self, d: RouteDecision) -> dict:
        return {"source": d.source, "target": d.target,
                "confidence": round(d.confidence, 4), "reason": d.reason,
                "manual": d.manual, "low_confidence": d.low_confidence}

    # ---- one utterance: translate + (optionally) speak, streaming ----
    def handle_utterance(self, audio: np.ndarray, emit=print, play: bool | None = None,
                         spec: Optional[_Speculation] = None) -> dict:
        if self.cfg.streaming_asr:
            return self.handle_utterance_streaming(audio, emit=emit, play=play, spec=spec)
        play = self.cfg.play if play is None else play
        source = self._source_before_asr()
        decision = self._decision_before_asr(source)
        if decision.source != "vi":
            raise RuntimeError(
                "--no-streaming-asr supports Vietnamese source only; use streaming ASR "
                f"for {decision.source}->{decision.target}"
            )
        target = decision.target
        emit(json.dumps({"type": "direction", **self._direction_event(decision)}))
        t0 = time.perf_counter()
        result = {"raw_vi": "", "vi": "", "tgt": "", "sr": SR,
                  "correction": None, "first_audio_ms": None}
        sd = None
        if play:
            try: import sounddevice as sd
            except Exception: sd = None
        for ev in self.pipe.run_streaming(audio, SR, target):
            st = ev["stage"]
            if st in ("asr_partial", "asr_done"):
                result["raw_vi"] = ev.get("raw_vi_text", ev["vi_text"])
                result["vi"] = ev["vi_text"]
                emit(json.dumps({"type": "partial", "field": "src", "text": ev["vi_text"]}))
            elif st == "correction_done":
                result["raw_vi"] = ev.get("raw_vi_text", result["raw_vi"])
                result["vi"] = ev["vi_text"]
                result["correction"] = ev.get("correction")
                emit(json.dumps({"type": "corrected", "raw": result["raw_vi"],
                                 "text": result["vi"],
                                 "correction": result["correction"],
                                 "latency_ms": ev.get("latency", {}).get("corrector_ms")}))
            elif st == "mt_partial":
                result["tgt"] = ev["tgt_text"]
                emit(json.dumps({"type": "partial", "field": "tgt", "text": ev["tgt_text"]}))
            elif st == "tts_partial":
                if result["first_audio_ms"] is None:
                    result["first_audio_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                if play and sd is not None and ev["tgt_audio"] is not None and len(ev["tgt_audio"]):
                    sd.play(ev["tgt_audio"], ev["sr"]); sd.wait()
            elif st in ("done", "cache_hit", "tm_full_hit"):
                result["vi"] = ev.get("vi_text", result["vi"])
                result["raw_vi"] = ev.get("raw_vi_text", result["raw_vi"] or result["vi"])
                result["correction"] = ev.get("correction", result["correction"])
                result["tgt"] = ev.get("tgt_text", result["tgt"])
                result["sr"] = ev.get("sr", SR)
                if play and sd is not None and not self.pipe.cfg.streaming_tts:
                    a = ev.get("tgt_audio")
                    if a is not None and len(a): sd.play(a, ev["sr"]); sd.wait()
            elif st == "vad_reject":
                emit(json.dumps({"type": "vad_reject", "reason": ev["tgt_text"]}))
                return result
        result["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["direction"] = self._direction_event(decision)
        emit(json.dumps({"type": "final", "raw_src": result["raw_vi"],
                         "src": result["vi"], "tgt": result["tgt"],
                         "correction": result["correction"],
                         "direction": result["direction"], "speculative": False,
                         "first_audio_ms": result["first_audio_ms"],
                         "total_ms": result["total_ms"]}))
        return result

    # ---- ASR helpers (shared by the speculative and the confirmed final path) ----
    def _router_asr_factory(self, source: str):
        """OnlineASR over whichever ASR model owns the preselected `source`.

        AsrRouter hands back a faster-whisper model, which is exactly what
        OnlineASR consumes, so the streaming machinery is untouched — only the
        weights and the forced language token change per turn.
        """
        lang = source
        if self._asr_router is None:
            from asr_router import AsrRouter
            self._asr_router = AsrRouter()
        if self._OnlineASR is None:
            from streaming_asr import OnlineASR
            self._OnlineASR = OnlineASR
        return self._OnlineASR(self._asr_router.model_for(lang), language=lang,
                               beam_size=self.pipe.cfg.asr_beam)

    def _make_online_asr(self, source: str):
        if self.deps.online_asr_factory is not None:
            asr = self.deps.online_asr_factory(source)
            if asr is not None:
                return asr
        if self.cfg.multilingual_asr:
            return self._router_asr_factory(source)
        if source != "vi":
            raise RuntimeError(
                f"source {source!r} requires --multilingual-asr; refusing to decode "
                "non-Vietnamese speech with PhoWhisper"
            )
        self.pipe._ensure_asr()
        if self._OnlineASR is None:
            from streaming_asr import OnlineASR
            self._OnlineASR = OnlineASR
        return self._OnlineASR(self.pipe._asr, language=self.pipe.cfg.asr_lang,
                               beam_size=self.pipe.cfg.asr_beam)

    def _transcribe_turn(self, audio: np.ndarray, source: str, emit=None) -> str:
        """Run one turn through the streaming ASR. `emit` is None for speculative
        passes: a speculation that gets discarded must never have shown text."""
        oa = self._make_online_asr(source)
        audio = np.asarray(audio, dtype=np.float32)
        step = max(1, int(self.cfg.asr_chunk * SR))
        last = ""
        for i in range(0, len(audio), step):
            oa.insert_audio(audio[i:i + step])
            oa.process_iter()
            committed = oa.full_text()
            if emit is not None and committed and committed != last:
                last = committed
                emit(json.dumps({"type": "partial", "field": "src", "text": committed}))
        oa.finish()
        return oa.full_text().strip()

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        if self._denoiser is None:
            return audio
        audio, _ = self._denoiser.denoise(audio, SR)
        return audio

    def _speculative_asr(self, audio: np.ndarray, source: str) -> str:
        audio = self._denoise(audio)
        with self._asr_lock:
            return self._transcribe_turn(audio, source, emit=None)

    def _start_speculation(self, audio: np.ndarray) -> _Speculation:
        source = self._source_before_asr()
        if self._spec_executor is None:
            self._spec_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="spec-asr")
        future = self._spec_executor.submit(self._speculative_asr, audio.copy(), source)
        return _Speculation(future, source)

    def _resolve_speculation(self, spec: Optional[_Speculation], source: str) -> Optional[str]:
        if spec is None or not spec.valid:
            return None
        if spec.source != source:
            spec.discard()
            return None
        try:
            # Waiting beats re-running: the pass is already in flight over the same
            # speech, and the ASR lock would serialize us behind it anyway.
            text = spec.future.result()
        except Exception as e:  # noqa: BLE001 — a bad speculation must never kill a turn
            print(f"[warn] speculative ASR failed, redoing: {e}", file=sys.stderr)
            return None
        return text if spec.valid else None

    # ---- MT/TTS pipelining ----
    def _translate_pipelined(self, sents: list, source: str, target: str):
        """Yield translations in order while the NEXT sentence is already being
        translated, so TTS of sentence i overlaps MT of sentence i+1.

        Deliberately one call per sentence instead of translate_sentences_batch:
        the batch call cannot return anything until the LAST sentence is decoded,
        which is exactly the serialization we are removing. For a single-sentence
        turn (the common case) the two are the same work.
        """
        out: "queue.Queue" = queue.Queue()

        def worker():
            try:
                for sent in sents:
                    out.put((self.pipe.translate(sent, target, source=source), None))
            except Exception as e:  # noqa: BLE001 — re-raised on the consumer thread
                out.put((None, e))
            finally:
                out.put(_EOS)

        threading.Thread(target=worker, name="mt-pipeline", daemon=True).start()
        while True:
            item = out.get()
            if item is _EOS:
                return
            text, err = item
            if err is not None:
                raise err
            yield text

    # ---- streaming-ASR turn: LocalAgreement-2 partials → MT → streaming TTS ----
    def handle_utterance_streaming(self, audio: np.ndarray, emit=print,
                                   play: bool | None = None,
                                   spec: Optional[_Speculation] = None) -> dict:
        """Within a completed VAD turn: take the speculative final if one is ready,
        otherwise run the (already-denoised) audio through OnlineASR in
        ~asr_chunk-second sub-chunks emitting stable LocalAgreement partials; then
        use the preselected direction, correct (Vietnamese source only), and run MT and
        streaming TTS pipelined sentence-by-sentence.

        Emits the same JSON events as handle_utterance (partial src/tgt,
        tts_partial, final) plus `direction`."""
        play = self.cfg.play if play is None else play
        t0 = time.perf_counter()
        result = {"raw_vi": "", "vi": "", "tgt": "", "sr": SR,
                  "correction": None, "first_audio_ms": None}
        sd = None
        if play:
            try: import sounddevice as sd
            except Exception: sd = None

        # The source pin is captured before ASR and owns the entire turn. Text LID
        # after decoding cannot repair an ASR pass made with the wrong language.
        source = self._source_before_asr()
        decision = self._decision_before_asr(source)
        result["direction"] = self._direction_event(decision)
        emit(json.dumps({"type": "direction", **result["direction"]}))

        # ----- ASR -----
        raw_src = self._resolve_speculation(spec, source)
        speculative = raw_src is not None
        if speculative:
            emit(json.dumps({"type": "partial", "field": "src", "text": raw_src}))
        else:
            # A segment handed over with a speculation was NOT denoised by the
            # dispatcher (the speculative worker denoised its own copy); if we are
            # here the speculation was discarded or failed, so pay it now.
            if spec is not None:
                audio = self._denoise(audio)
            with self._asr_lock:
                raw_src = self._transcribe_turn(audio, source, emit=emit)
        result["raw_vi"] = raw_src

        t_corrector = time.perf_counter()
        if decision.source == "vi":
            vi, correction = self.pipe.finalize_transcript(raw_src)
        else:
            vi, correction = raw_src, None
        correction_ms = round((time.perf_counter() - t_corrector) * 1000, 1)
        result["vi"] = vi
        result["correction"] = correction
        emit(json.dumps({"type": "corrected", "raw": raw_src, "text": vi,
                         "correction": correction, "latency_ms": correction_ms,
                         "speculative": speculative}))

        # ----- MT → streaming TTS, pipelined -----
        pb = self._get_playback() if (play and sd is not None) else None
        if pb is not None:
            pb.new_response()   # clear any prior barge-in interrupt
        sents = self._split_sentences(vi)
        spoken: list = []
        frag_index = 0
        for tgt_sent in self._translate_pipelined(sents, decision.source, decision.target):
            if not tgt_sent:
                continue
            spoken.append(tgt_sent)
            result["tgt"] = " ".join(spoken).strip()
            emit(json.dumps({"type": "partial", "field": "tgt", "text": result["tgt"]}))
            # Per-sentence stream so streaming_tts keeps its deliberately short
            # FIRST fragment — the fast-first-audio path from Round 4.
            for frag in self.stts.stream(tgt_sent, decision.target):
                if result["first_audio_ms"] is None:
                    result["first_audio_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                emit(json.dumps({"type": "tts_partial", "play": True,
                                 "index": frag_index, "text": frag.text}))
                frag_index += 1
                if pb is not None and frag.audio is not None and len(frag.audio):
                    pb.enqueue(frag.audio, frag.sr)
        # File/batch mode drains fully here; the live mic loop lets playback run in the
        # background (half-duplex reopens the mic when it drains, see _reopen_mic).
        if pb is not None:
            pb.end_response()
            if self._sync_playback:
                pb.wait_idle()

        result["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        emit(json.dumps({"type": "final", "raw_src": result["raw_vi"],
                         "src": result["vi"], "tgt": result["tgt"],
                         "correction": result["correction"],
                         "direction": result["direction"], "speculative": speculative,
                         "first_audio_ms": result["first_audio_ms"],
                         "total_ms": result["total_ms"]}))
        return result

    def _dispatch_segment(self, seg: np.ndarray, emit, play,
                          spec: Optional[_Speculation] = None):
        # BARGE-IN (opt-in only): a new turn arrived while we're still speaking the
        # previous one → stop the old output. Mutually exclusive with half-duplex,
        # which keeps the mic shut instead so this can never trigger.
        if self.cfg.barge_in and play and not self._sync_playback \
                and self._playback is not None and self._playback.is_speaking():
            dropped = self._playback.interrupt()
            emit(json.dumps({"type": "barge_in", "dropped_chunks": dropped}))
        # Denoise the whole utterance ONCE, if it is enabled at all -- see the note on
        # LiveConfig.denoise for why the default is off and why this is the wrong
        # place for GTCRN to earn its keep.
        if self.cfg.denoise and self._denoiser is not None and spec is None:
            seg, _ = self._denoiser.denoise(seg, SR)
        self.handle_utterance(seg, emit=emit, play=play, spec=spec)

    # ---- half-duplex mic gate ----
    def _half_duplex_active(self, play: bool) -> bool:
        # Only meaningful when our own speaker is live in the same room as the mic:
        # file/sidecar runs have no speaker in the loop and must never drop input.
        return bool(self.cfg.half_duplex and not self.cfg.barge_in
                    and play and not self._sync_playback)

    def _close_mic(self, emit) -> None:
        self._mic_open.clear()
        emit(json.dumps({"type": "mic_gate", "open": False, "reason": "turn_detected"}))

    def _reopen_mic(self, emit) -> None:
        # RTranslator restarts the recorder in the TTS-completion callback, not when
        # the text is ready; plus RealtimeSTT's min_gap so our own decaying tail
        # cannot retrigger a turn.
        if self._playback is not None:
            try: self._playback.wait_idle()
            except Exception: pass
        if self.cfg.min_gap_between_turns > 0:
            time.sleep(self.cfg.min_gap_between_turns)
        self._mic_open.set()
        emit(json.dumps({"type": "mic_gate", "open": True, "reason": "playback_done"}))

    # ---- feed a stream of audio through VAD, dispatch each completed turn ----
    def _feed_frames(self, frame_iter, emit=print, play=True, drop_oldest=False):
        """Capture thread → frame queue → this thread (VAD) → turn queue → worker.

        `drop_oldest` is the realtime (microphone) policy: when processing falls
        behind, discard the OLDEST queued frames so latency stays bounded. File and
        sidecar input is not realtime — there the queue blocks, because dropping
        would silently corrupt a deterministic run.
        """
        emit_lock = threading.Lock()

        def out(msg):   # three threads share one JSON-lines stream
            with emit_lock:
                emit(msg)

        gated = self._half_duplex_active(play)
        frame_q: "queue.Queue" = queue.Queue(maxsize=self.cfg.frame_queue_max)
        turn_q: "queue.Queue" = queue.Queue(maxsize=self.cfg.turn_queue_max)
        self.frames_dropped = 0
        self.frames_gated = 0
        self._mic_open.set()
        stop = threading.Event()
        last_drop_log = [0.0]

        def capture():
            try:
                for frame in frame_iter:
                    if stop.is_set():
                        break
                    if gated and not self._mic_open.is_set():
                        self.frames_gated += 1
                        continue
                    arr = np.asarray(frame, dtype=np.float32)
                    if drop_oldest:
                        n = _put_drop_oldest(frame_q, arr)
                        if n:
                            self.frames_dropped += n
                            # Emitted from the capture thread so the drop is
                            # timestamped when it happens; silent dropping is a
                            # worse bug than the backlog it fixes.
                            now = time.perf_counter()
                            if now - last_drop_log[0] > 1.0:
                                last_drop_log[0] = now
                                print(f"[warn] input backlog: dropped {self.frames_dropped} "
                                      f"frames (queue={self.cfg.frame_queue_max})",
                                      file=sys.stderr)
                            out(json.dumps({"type": "frames_dropped", "count": n,
                                            "total": self.frames_dropped}))
                    else:
                        frame_q.put(arr)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] capture stopped: {e}", file=sys.stderr)
            finally:
                frame_q.put(_EOS)

        def process():
            while True:
                item = turn_q.get()
                if item is _EOS:
                    return
                seg, spec = item
                try:
                    self._dispatch_segment(seg, out, play, spec=spec)
                except Exception as e:  # noqa: BLE001 — one bad turn must not end the session
                    print(f"[warn] turn failed: {e}", file=sys.stderr)
                    out(json.dumps({"type": "turn_error", "error": str(e)}))
                finally:
                    if gated:
                        self._reopen_mic(out)

        vad = self._new_vad()
        buf = np.zeros(0, dtype=np.float32)
        # Two aligned streams: `buf` feeds the VAD, `raw_buf` feeds ASR. When the
        # streaming denoiser is on they differ in content but are kept the same
        # length, so a window consumed from one always has its twin in the other.
        # They are offset by the model's priming delay (~128 samples, 8 ms), which
        # is far inside the 300 ms pre-roll and needs no correction.
        raw_buf = np.zeros(0, dtype=np.float32)
        raw_pending = np.zeros(0, dtype=np.float32)
        stream_denoiser = None
        if self.cfg.denoise_vad:
            try:
                from edge_frontend import make_streaming_denoiser
                stream_denoiser = make_streaming_denoiser()
            except Exception as exc:                      # noqa: BLE001
                print(f"[warn] streaming denoiser unavailable, VAD sees raw audio: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)

        # --- pre-roll ring buffer + short-turn guard (from KoljaB/RealtimeSTT) ---
        # 1) PRE-ROLL: Silero fires a few frames LATE, so the first phoneme of a turn
        #    is usually clipped — costly for short Vietnamese words. We keep a rolling
        #    history of the most recent input samples and, when a turn closes, splice
        #    in `preroll_ms` of audio captured *just before* the detected onset
        #    (vad.front.start) so the onset is never clipped.
        # 2) SHORT-TURN GUARD: drop turns whose *speech* span is < min_speech_ms so we
        #    never pay ASR→MT→TTS on noise blips/clicks.
        # 3) SPECULATIVE FINAL: once Silero stops reporting speech we are in the
        #    min_silence countdown. `early_transcription_s` into it we fire the final
        #    ASR on the accumulated turn audio, so the result is (usually) ready the
        #    instant sherpa declares the turn over — hiding the endpoint wait and most
        #    of ASR from perceived latency. If speech resumes the speculation is
        #    discarded and never emitted (RealtimeSTT allowed_to_early_transcribe).
        preroll_n = int(self.cfg.preroll_ms / 1000 * SR)
        min_speech_n = int(self.cfg.min_speech_ms / 1000 * SR)
        # With denoise_vad on, `history` is also the source of the turn audio itself,
        # not just the pre-roll, so it has to outlive the longest turn we accept.
        hist_max = preroll_n + int((self.cfg.min_silence + 4.0) * SR)
        if self.cfg.denoise_vad:
            hist_max = preroll_n + int((self.cfg.min_silence + 30.0) * SR)
        history = np.zeros(0, dtype=np.float32)
        fed = 0  # absolute count of samples handed to the VAD (matches front.start frame)
        turn_audio = np.zeros(0, dtype=np.float32)
        speaking = False
        silence_start: Optional[float] = None
        spec: Optional[_Speculation] = None

        def take(seg_obj):
            seg = np.asarray(seg_obj.samples, dtype=np.float32)
            # guard measured on the raw speech span, BEFORE pre-roll is prepended
            if self.cfg.early_finalize and min_speech_n > 0 and len(seg) < min_speech_n:
                return None
            start = getattr(seg_obj, "start", None)
            if stream_denoiser is not None and start is not None:
                # The VAD segments the DENOISED stream, so its own samples are
                # denoised too — and those must never reach the recogniser. `history`
                # holds the raw audio on the same absolute timeline (both streams are
                # appended in lockstep), so re-cut the segment from there.
                base = fed - len(history)
                lo = max(0, min(start - base, len(history)))
                hi = max(lo, min(start - base + len(seg), len(history)))
                if hi > lo:
                    seg = history[lo:hi].copy()
            if preroll_n > 0 and start is not None:
                hist_start = fed - len(history)          # absolute index of history[0]
                lo = max(0, min(start - preroll_n - hist_start, len(history)))
                hi = max(0, min(start - hist_start, len(history)))
                if hi > lo:
                    seg = np.concatenate([history[lo:hi], seg])
            return seg

        def drain():
            nonlocal spec, turn_audio, silence_start
            while not vad.empty():
                seg = take(vad.front)
                vad.pop()
                turn_audio = np.zeros(0, dtype=np.float32)
                silence_start = None
                if seg is None:
                    if spec is not None:
                        spec.discard()
                        spec = None
                    continue
                if gated:
                    # RTranslator stops the recorder outright at turn detection;
                    # queued frames are the tail of the turn we are about to speak.
                    self._close_mic(out)
                    self.frames_gated += _drop_frames(frame_q)
                turn_q.put((seg, spec))
                spec = None

        cap = threading.Thread(target=capture, name="capture", daemon=True)
        proc = threading.Thread(target=process, name="turn-worker", daemon=True)
        cap.start()
        proc.start()
        was_gated = False
        try:
            while True:
                frame = frame_q.get()
                if frame is _EOS:
                    break
                if gated and not self._mic_open.is_set():
                    self.frames_gated += 1
                    was_gated = True
                    continue
                if was_gated:
                    # Fresh VAD state after our own speech: anything half-accumulated
                    # before the gate closed belongs to the turn we just spoke.
                    was_gated = False
                    if hasattr(vad, "reset"):
                        vad.reset()
                    buf = np.zeros(0, dtype=np.float32)
                    raw_buf = np.zeros(0, dtype=np.float32)
                    raw_pending = np.zeros(0, dtype=np.float32)
                    if stream_denoiser is not None:
                        stream_denoiser.reset()
                    turn_audio = np.zeros(0, dtype=np.float32)
                    speaking = False
                    silence_start = None
                frame = frame.astype(np.float32)
                if stream_denoiser is None:
                    vad_new = raw_new = frame
                else:
                    vad_new = stream_denoiser.accept(frame, SR)
                    raw_pending = np.concatenate([raw_pending, frame])
                    n_aligned = min(len(vad_new), len(raw_pending))
                    raw_new = raw_pending[:n_aligned]
                    raw_pending = raw_pending[n_aligned:]
                    vad_new = vad_new[:n_aligned]
                buf = np.concatenate([buf, vad_new])
                raw_buf = np.concatenate([raw_buf, raw_new])
                while len(buf) >= self._window:
                    chunk = buf[:self._window]
                    vad.accept_waveform(chunk)
                    buf = buf[self._window:]
                    # ASR must never see the denoised copy: the same T4 grid found
                    # GTCRN neutral-to-harmful in front of the recogniser.
                    raw_chunk, raw_buf = raw_buf[:self._window], raw_buf[self._window:]
                    history = np.concatenate([history, raw_chunk])
                    fed += len(chunk)
                    if len(history) > hist_max:
                        history = history[len(history) - hist_max:]

                    now_speaking = bool(vad.is_speech_detected())
                    if now_speaking:
                        if not speaking:
                            if spec is not None:   # speech resumed → discard, never emit
                                spec.discard()
                                spec = None
                            lo = max(0, len(history) - len(raw_chunk) - preroll_n)
                            turn_audio = history[lo:len(history) - len(raw_chunk)].copy()
                        turn_audio = np.concatenate([turn_audio, raw_chunk])
                        silence_start = None
                    elif speaking:
                        silence_start = time.perf_counter()
                    elif (silence_start is not None and spec is None
                          and self.cfg.speculative_final and len(turn_audio) >= min_speech_n
                          and self.router.manual_source is not None
                          and time.perf_counter() - silence_start
                          >= self.cfg.early_transcription_s):
                        spec = self._start_speculation(turn_audio)
                        out(json.dumps({"type": "speculative_final",
                                        "samples": int(len(turn_audio))}))
                    speaking = now_speaking
                    drain()
                    if gated and not self._mic_open.is_set():
                        was_gated = True   # turn dispatched: drop this frame's tail too
                        break
            vad.flush()
            drain()
        finally:
            stop.set()
            _drain_queue(frame_q)   # unblock a capture thread parked on a full queue
            turn_q.put(_EOS)
            proc.join()
            if spec is not None:
                spec.discard()

    def run_mic(self, emit=print):
        import sounddevice as sd
        step = int(0.1 * SR)
        self._sync_playback = False   # background playback; the mic gate reopens on drain
        mode = "barge-in" if self.cfg.barge_in else "half-duplex"
        print(f"[live] listening… pair={self.cfg.pair[0]}<->{self.cfg.pair[1]} "
              f"source={self.router.manual_source} {mode}  (Ctrl-C to stop)",
              file=sys.stderr)

        def frames():
            with sd.InputStream(channels=1, dtype="float32", samplerate=SR) as stream:
                while True:
                    data, _ = stream.read(step)
                    yield data.reshape(-1)

        try:
            self._feed_frames(frames(), emit=emit, play=self.cfg.play, drop_oldest=True)
        except KeyboardInterrupt:
            print("\n[live] stopped", file=sys.stderr)

    def run_file(self, wav: str, emit=print, play=False):
        import soundfile as sf
        audio, sr = sf.read(wav, dtype="float32", always_2d=False)
        if audio.ndim > 1: audio = audio[:, 0]
        if sr != SR:
            n = int(round(len(audio) * SR / sr))
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        step = int(0.1 * SR)
        frames = (audio[i:i + step] for i in range(0, len(audio), step))
        self._feed_frames(frames, emit=emit, play=play)

    def run_sidecar(self, emit=print, control=None):
        """Read raw s16le/16k/mono PCM from stdin; emit JSON events to stdout.

        `control` is an optional newline-delimited JSON stream (a file, FIFO or any
        iterable of lines) carrying direction commands from the host — this is what
        a physical two-button push-to-talk drives:
            {"type": "set_direction", "source": "vi"}
            {"type": "clear_direction"}
        It is a SEPARATE stream on purpose: stdin carries raw PCM and interleaving
        text into it would desynchronise the frame boundaries.
        """
        step = int(0.1 * SR) * 2  # bytes (int16)
        stdin = sys.stdin.buffer

        if control is not None:
            threading.Thread(target=self._control_loop, args=(control, emit),
                             name="control", daemon=True).start()

        def frames():
            while True:
                chunk = stdin.read(step)
                if not chunk:
                    break
                yield (np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)

        self._feed_frames(frames(), emit=emit, play=False)

    def _control_loop(self, control, emit=print):
        for line in control:
            line = (line or "").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                kind = msg.get("type")
                if kind == "set_direction":
                    self.set_manual_source(msg.get("source"))
                elif kind == "clear_direction":
                    self.set_manual_source(None)
                else:
                    raise ValueError(f"unknown control message {kind!r}")
            except Exception as e:  # noqa: BLE001 — a bad control line must not kill audio
                emit(json.dumps({"type": "control_error", "error": str(e), "line": line}))
                continue
            emit(json.dumps({"type": "direction_pinned", **self.direction_state()}))


def _drop_frames(q: "queue.Queue") -> int:
    """Discard queued frames, KEEPING the end-of-stream sentinel — swallowing it
    would strand the reader forever."""
    n = 0
    saw_eos = False
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is _EOS:
            saw_eos = True
        else:
            n += 1
    if saw_eos:
        q.put(_EOS)
    return n


def _drain_queue(q: "queue.Queue") -> int:
    n = 0
    while True:
        try:
            q.get_nowait()
            n += 1
        except queue.Empty:
            return n


def _resolve_cli_denoise(value: Optional[bool]) -> bool:
    """Preserve the environment default unless the CLI explicitly overrides it."""
    return LiveConfig().denoise if value is None else bool(value)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Live on-device speech translator")
    ap.add_argument("mode", choices=["mic", "file", "sidecar"])
    ap.add_argument("--wav", help="input wav (file mode)")
    ap.add_argument("--target", default="en", choices=["vi", "en", "zh", "ko"],
                    help="the non-Vietnamese side of the pair (back-compat spelling of --pair)")
    ap.add_argument("--pair", help="configured bidirectional pair, e.g. vi,ko")
    ap.add_argument("--source", "--manual-source", dest="manual_source",
                    required=True, choices=["vi", "en", "zh", "ko"],
                    help="initial source language selected before ASR; use the alias "
                         "--manual-source for existing launch scripts")
    ap.add_argument("--control", help="sidecar mode: newline-delimited JSON control "
                                      "stream (file or FIFO) for direction commands")
    denoise_group = ap.add_mutually_exclusive_group()
    denoise_group.add_argument(
        "--denoise", dest="denoise", action="store_true",
        help="opt in to whole-utterance GTCRN before ASR (disabled by default; "
             "DEMAND evidence shows significant harm in 5/12 cells)")
    denoise_group.add_argument(
        "--no-denoise", dest="denoise", action="store_false",
        help="explicitly keep GTCRN off (backward-compatible flag)")
    ap.set_defaults(denoise=None)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--no-streaming-asr", action="store_true",
                    help="disable LocalAgreement-2 streaming ASR; use whole-utterance run_streaming")
    ap.add_argument("--preroll-ms", type=int, default=300,
                    help="pre-roll spliced before each turn to recover clipped onsets (0=off)")
    ap.add_argument("--no-early-finalize", action="store_true",
                    help="disable the short-turn guard (dispatch even ultra-short VAD turns)")
    ap.add_argument("--min-speech-ms", type=int, default=200,
                    help="drop VAD turns whose speech span is shorter than this (ms)")
    ap.add_argument("--min-silence", type=float, default=0.5,
                    help="end a turn after this much trailing silence (s)")
    ap.add_argument("--no-speculative", action="store_true",
                    help="disable the speculative final ASR fired during the endpoint countdown")
    ap.add_argument("--early-transcription", type=float, default=0.25,
                    help="fire the speculative final this far into the silence countdown (s)")
    ap.add_argument("--multilingual-asr", action="store_true",
                    help="route en/zh/ko turns to a multilingual Whisper instead of "
                         "PhoWhisper; adds a second resident ASR model and its zh/ko "
                         "accuracy is unmeasured")
    ap.add_argument("--barge-in", action="store_true",
                    help="keep the mic live during playback so a new turn interrupts it; "
                         "disables the half-duplex guard (own speaker WILL be re-captured)")
    ap.add_argument("--no-half-duplex", action="store_true",
                    help="do not close the mic while speaking (no barge-in either)")
    ap.add_argument("--min-gap", type=float, default=0.3,
                    help="minimum silence after our own playback before a new turn (s)")
    ap.add_argument("--frame-queue", type=int, default=50,
                    help="capture queue depth in 100 ms frames (drop-oldest above this)")
    ap.add_argument("--no-correct", action="store_true",
                    help="disable final Vietnamese transcript correction")
    ap.add_argument("--corrector-model", default="bmd1905/vietnamese-correction-v2",
                    help="local Hugging Face correction model prepared in advance")
    ap.add_argument("--lazy-load", action="store_true",
                    help="skip startup prewarm (first turn will be much slower)")
    args = ap.parse_args()

    pair = tuple(args.pair.replace("/", ",").split(",")) if args.pair else None
    try:
        cfg = LiveConfig(target=args.target, pair=pair, manual_source=args.manual_source,
                         denoise=_resolve_cli_denoise(args.denoise), play=not args.no_play,
                         streaming_asr=not args.no_streaming_asr,
                         multilingual_asr=args.multilingual_asr,
                         preroll_ms=args.preroll_ms,
                         early_finalize=not args.no_early_finalize,
                         min_speech_ms=args.min_speech_ms, min_silence=args.min_silence,
                         speculative_final=not args.no_speculative,
                         early_transcription_s=args.early_transcription,
                         half_duplex=not args.no_half_duplex, barge_in=args.barge_in,
                         min_gap_between_turns=args.min_gap,
                         frame_queue_max=args.frame_queue,
                         corrector=not args.no_correct,
                         corrector_model=args.corrector_model,
                         prewarm=not args.lazy_load)
    except ValueError as exc:
        ap.error(str(exc))
    lt = LiveTranslator(cfg)
    if args.mode == "mic":
        lt.run_mic()
    elif args.mode == "file":
        if not args.wav:
            ap.error("--wav required for file mode")
        lt.run_file(args.wav, play=not args.no_play)
    else:
        ctl = open(args.control, "r", encoding="utf-8") if args.control else None
        lt.run_sidecar(control=ctl)
