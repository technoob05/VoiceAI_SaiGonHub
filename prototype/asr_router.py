"""Two-model ASR routing: keep the Vietnamese specialist, add a multilingual model.

The competition is bidirectional over three pairs (vi<->en, vi<->zh, vi<->ko), but
the ASR the pipeline ships today is PhoWhisper-small INT8 -- a Vietnamese-specialised
Whisper. Fed English, Mandarin or Korean speech it does not fail loudly; it emits
confident Vietnamese-looking tokens, which is exactly the failure mode lang_router's
docstring warns about ("a Vietnamese-only ASR fed Korean speech emits Vietnamese-
looking tokens"). So the inbound half of five of the six directions is simply unsolved.

Replacing PhoWhisper with generic Whisper would trade one hole for another: the
tone-quantization research spine is measured on PhoWhisper, and PhoWhisper is
materially better on Vietnamese than a same-size generic Whisper. The answer is
therefore two resident models plus a router, not a substitution:

    vi              -> PhoWhisper-small INT8      (the specialist, unchanged)
    en / zh / ko    -> faster-whisper-small       (multilingual, INT8 at load time)

Design constraints this module is built around:

  * ALWAYS-WARM. A conversation device cannot afford a mid-turn model load, so a
    model is never unloaded once resident. `prewarm()` pays the load plus one dummy
    inference upfront for exactly the languages a session needs.
  * LAZY. A Vietnamese-only session must not pay ~500 MB of RAM and a multi-second
    load for a multilingual model it will never call. Loading is deferred to first
    use of a language that needs it.
  * FORCED LANGUAGE TOKEN. The caller has already decided the direction; Whisper's
    own language detection is a second, redundant, and less reliable decision. Every
    decode pins `language=` explicitly, and the result records what was pinned.
  * ATTRIBUTABLE. Each result says which model served it and how long it took, so a
    benchmark can price a mixed conversation instead of guessing.

COMPOSITION WITH lang_router
    This module owns no direction logic. The host or physical push-to-talk control
    must select the source before ASR. `transcribe_route()` accepts only an explicitly
    pinned RouteDecision; transcript-based routing happens too late to select the
    acoustic model and is rejected.

WHAT THIS CANNOT DO -- read before trusting it
  * ACCURACY ON zh/ko IS UNMEASURED. This module makes the en/zh/ko path *exist*.
    Whether generic whisper-small is good enough on Mandarin and Korean for this use
    case is an open question: the repo has no labelled zh/ko test audio, so nothing
    here has been scored against a reference. Latency and memory below are measured;
    word error rate is not. Do not cite an accuracy number that was never run.
  * NO ACOUSTIC LANGUAGE ID. If the caller pins the wrong language, this module will
    faithfully decode as the wrong language. Detecting the language *from audio* is a
    separate stage neither this module nor lang_router provides.
  * Two resident models is two models' worth of RAM. `footprint()` reports the real
    on-disk cost so the budget is stated rather than assumed.
"""

from __future__ import annotations

import os
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CT2_ROOT = PROJECT / "data" / "raw" / "models_ct2"

SAMPLE_RATE = 16_000

ROLE_VI = "vi_specialist"
ROLE_MULTI = "multilingual"

#: Which resident model serves which language. Vietnamese is the only language the
#: specialist is trained for; everything else in the competition set goes multilingual.
LANG_TO_ROLE: dict[str, str] = {
    "vi": ROLE_VI,
    "en": ROLE_MULTI,
    "zh": ROLE_MULTI,
    "ko": ROLE_MULTI,
}

SUPPORTED = tuple(LANG_TO_ROLE)

#: Whisper's own language tokens. They happen to match our two-letter codes for all
#: four competition languages, but the mapping is explicit so a future code (e.g.
#: "zh-Hant") cannot silently be passed through as a Whisper token.
WHISPER_LANG = {"vi": "vi", "en": "en", "zh": "zh", "ko": "ko"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _dir_size_bytes(path: Path) -> int:
    """On-disk weight of a model directory, ignoring the HF download cache.

    `snapshot_download(local_dir=...)` leaves a `.cache/huggingface` tree of blob
    metadata next to the weights. Counting it would overstate the deployed cost by
    roughly the size of the weights again, so it is excluded.
    """
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not any(part.startswith(".") for part in item.relative_to(path).parts):
            total += item.stat().st_size
    return total


def normalize_lang(lang: str) -> str:
    code = (lang or "").strip().lower()
    if code not in LANG_TO_ROLE:
        raise ValueError(f"unsupported ASR language {lang!r}; supported: {SUPPORTED}")
    return code


@dataclass
class AsrRouterConfig:
    """Paths and decoding knobs for both resident models.

    Defaults mirror `pipeline_fast.FastPipelineConfig` so a transcript produced
    through this router on Vietnamese is decoded identically to one produced by the
    existing pipeline -- the router must not silently change Vietnamese behaviour.
    """

    vi_model_dir: str = _env_str(
        "VOICEAI_ASR_VI_DIR", str(CT2_ROOT / "phowhisper-small-int8")
    )
    multi_model_dir: str = _env_str(
        "VOICEAI_ASR_MULTI_DIR", str(CT2_ROOT / "whisper-small-multilingual")
    )

    device: str = _env_str("VOICEAI_ASR_DEVICE", "cpu")
    compute_type: str = _env_str("VOICEAI_ASR_COMPUTE_TYPE", "int8")

    cpu_threads: int = int(os.environ.get("VOICEAI_CT2_INTRA_THREADS", 0)) or max(
        (os.cpu_count() or 4) - 2, 4
    )
    num_workers: int = 1

    beam_size: int = 1
    temperature: float = 0.0
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2
    compression_ratio_threshold: float = 1.8

    #: Domain priming, Vietnamese only -- the specialist is the model with a known
    #: in-domain vocabulary. Priming a multilingual decoder with Vietnamese text
    #: would bias it toward the wrong language, which is the opposite of the point.
    vi_initial_prompt: str = _env_str("VOICEAI_ASR_PROMPT", "")

    #: Whisper emits Traditional Chinese for a third of Mandarin utterances, which
    #: is not a hearing error but an orthography choice -- and a costly one here,
    #: because MT is configured `zho_Hans`, the glossary's Mandarin column is
    #: Simplified and the Mandarin voice expects Simplified input, so a Traditional
    #: transcript degrades every stage downstream. Priming the decoder with a
    #: Simplified prompt measured CER 0.2311 -> 0.1324 (-42.7% relative) and
    #: Traditional rows 10/30 -> 2/30 on FLEURS cmn_hans_cn (n=30, CPU INT8).
    #: Set VOICEAI_ASR_PROMPT_ZH="" to disable.
    zh_initial_prompt: str = _env_str("VOICEAI_ASR_PROMPT_ZH",
                                      "以下是普通话的句子，请使用简体中文。")

    def dir_for_role(self, role: str) -> str:
        if role == ROLE_VI:
            return self.vi_model_dir
        if role == ROLE_MULTI:
            return self.multi_model_dir
        raise ValueError(f"unknown model role {role!r}")


@dataclass
class ModelInfo:
    """What a resident model costs and how much work it has done."""

    role: str
    path: str
    loaded: bool
    exists_on_disk: bool
    disk_bytes: int
    compute_type: str
    load_ms: Optional[float] = None
    prewarm_ms: Optional[float] = None
    calls: int = 0
    total_decode_ms: float = 0.0

    @property
    def disk_mb(self) -> float:
        return self.disk_bytes / 1e6

    def to_dict(self) -> dict:
        out = asdict(self)
        out["disk_mb"] = round(self.disk_mb, 1)
        return out


@dataclass
class TranscriptionResult:
    """One decode, with the attribution a benchmark needs.

    `role` and `model_path` answer "who served this", `latency_ms` answers "what did
    it cost", and `forced_language` records the token that was pinned -- so a wrong
    transcript can be traced to a wrong pin rather than blamed on the model.
    """

    text: str
    lang: str
    role: str
    model_path: str
    forced_language: str
    latency_ms: float
    audio_seconds: float
    load_ms: float = 0.0
    segments: int = 0
    language_probability: Optional[float] = None

    @property
    def rtf(self) -> float:
        """Real-time factor: decode seconds per audio second. <1 is faster than real time."""
        if self.audio_seconds <= 0:
            return float("nan")
        return (self.latency_ms / 1000.0) / self.audio_seconds

    def to_dict(self) -> dict:
        out = asdict(self)
        out["rtf"] = round(self.rtf, 3)
        return out


ModelLoader = Callable[[str, AsrRouterConfig], Any]


def default_loader(model_dir: str, cfg: AsrRouterConfig) -> Any:
    """Construct a faster-whisper WhisperModel, matching pipeline_fast's construction.

    faster-whisper repos on HuggingFace already ship CTranslate2 weights, so INT8 is
    produced here at load time via `compute_type` -- there is no conversion step.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_dir,
        device=cfg.device,
        compute_type=cfg.compute_type,
        cpu_threads=cfg.cpu_threads,
        num_workers=cfg.num_workers,
    )


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample, identical to pipeline_fast.transcribe_raw's path."""
    audio = np.asarray(audio, dtype=np.float32)
    if sr == SAMPLE_RATE:
        return audio
    target_len = int(round(len(audio) * SAMPLE_RATE / sr))
    return np.interp(
        np.linspace(0, len(audio), target_len, endpoint=False),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def load_wav_mono16k(path: str | Path) -> np.ndarray:
    """Read a PCM wav to float32 mono 16 kHz without pulling in soundfile/librosa."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width {width} bytes in {path}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return resample_to_16k(data, sr)


class AsrRouter:
    """Owns two lazily-loaded, never-unloaded faster-whisper models and routes by language.

    Residency is the contract: once a model is loaded it stays, because a reload in
    the middle of a conversation is a multi-second dead air the device cannot spend.
    Laziness is the other half: a session that only ever speaks Vietnamese never
    constructs the multilingual model at all.
    """

    def __init__(
        self,
        config: Optional[AsrRouterConfig] = None,
        loader: Optional[ModelLoader] = None,
        verbose: bool = True,
    ):
        self.cfg = config or AsrRouterConfig()
        self._loader = loader or default_loader
        self.verbose = verbose
        self._models: dict[str, Any] = {}
        self._info: dict[str, ModelInfo] = {
            role: self._fresh_info(role) for role in (ROLE_VI, ROLE_MULTI)
        }

    # ---------------- introspection ----------------

    def _fresh_info(self, role: str) -> ModelInfo:
        path = Path(self.cfg.dir_for_role(role))
        return ModelInfo(
            role=role,
            path=str(path),
            loaded=False,
            exists_on_disk=path.exists(),
            disk_bytes=_dir_size_bytes(path),
            compute_type=self.cfg.compute_type,
        )

    @staticmethod
    def role_for(lang: str) -> str:
        """Which resident model serves `lang`. Pure lookup, no side effects."""
        return LANG_TO_ROLE[normalize_lang(lang)]

    def is_loaded(self, role: str) -> bool:
        return role in self._models

    def loaded_roles(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def model_info(self, role: str) -> ModelInfo:
        return self._info[role]

    def footprint(self) -> dict:
        """Real on-disk cost of what is resident, and of what would be if fully loaded.

        Reported from disk rather than from RSS on purpose: process RSS on Windows
        mixes in the interpreter, numpy, and the MT/TTS models, so it cannot be
        attributed per-ASR-model. Disk weight of the CTranslate2 directory is the
        honest, attributable proxy for the INT8 weight budget. It is a proxy, not a
        measurement of RAM: runtime overhead (activations, KV cache) sits on top.

        One asymmetry worth stating out loud: PhoWhisper ships already-quantized INT8
        (~252 MB on disk), while `Systran/faster-whisper-small` ships float16 (~486 MB
        on disk) and is quantized to INT8 by CTranslate2 at load time. So the
        multilingual model's disk figure OVERSTATES its resident weight by roughly 2x;
        the two models are in fact comparable in RAM. Disk is the number that can be
        measured honestly here, so disk is the number reported.
        """
        per_model = {role: info.to_dict() for role, info in self._info.items()}
        resident = sum(i.disk_bytes for i in self._info.values() if i.loaded)
        both = sum(i.disk_bytes for i in self._info.values())
        return {
            "models": per_model,
            "loaded_roles": list(self.loaded_roles()),
            "resident_disk_bytes": resident,
            "resident_disk_mb": round(resident / 1e6, 1),
            "both_models_disk_bytes": both,
            "both_models_disk_mb": round(both / 1e6, 1),
        }

    # ---------------- loading ----------------

    def _ensure(self, role: str) -> Any:
        if role in self._models:
            return self._models[role]

        model_dir = self.cfg.dir_for_role(role)
        path = Path(model_dir)
        if not path.exists():
            hint = (
                "huggingface_hub.snapshot_download('Systran/faster-whisper-small', "
                f"local_dir='{model_dir}')"
                if role == ROLE_MULTI
                else "python -m ctranslate2.converters.transformers --model vinai/PhoWhisper-small "
                f"--output_dir {model_dir} --quantization int8"
            )
            raise FileNotFoundError(
                f"ASR model for role {role!r} not found at {model_dir}. Get it with: {hint}"
            )

        if self.verbose:
            print(
                f"[load] ASR {role}: {model_dir}  compute={self.cfg.compute_type}  "
                f"threads={self.cfg.cpu_threads}"
            )
        t0 = time.perf_counter()
        model = self._loader(model_dir, self.cfg)
        load_ms = (time.perf_counter() - t0) * 1000.0

        self._models[role] = model
        info = self._info[role]
        info.loaded = True
        info.load_ms = load_ms
        info.exists_on_disk = True
        info.disk_bytes = _dir_size_bytes(path)
        if self.verbose:
            print(f"[load] ASR {role} ready in {load_ms:.0f} ms ({info.disk_mb:.0f} MB on disk)")
        return model

    def model_for(self, lang: str) -> Any:
        """The model that serves `lang`, loading it on first use and keeping it forever."""
        return self._ensure(self.role_for(lang))

    # ---------------- decoding ----------------

    def _decode_kwargs(self, lang: str) -> dict:
        kwargs = dict(
            language=WHISPER_LANG[lang],
            task="transcribe",
            beam_size=self.cfg.beam_size,
            temperature=self.cfg.temperature,
            vad_filter=self.cfg.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=self.cfg.vad_min_silence_ms),
            condition_on_previous_text=False,
            without_timestamps=True,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            repetition_penalty=self.cfg.repetition_penalty,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
        )
        if lang == "vi" and self.cfg.vi_initial_prompt:
            kwargs["initial_prompt"] = self.cfg.vi_initial_prompt
        elif lang == "zh" and self.cfg.zh_initial_prompt:
            kwargs["initial_prompt"] = self.cfg.zh_initial_prompt
        return kwargs

    def transcribe(
        self,
        audio: np.ndarray,
        lang: str,
        sr: int = SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Decode `audio` as `lang`, on whichever model owns that language.

        The language token is always forced. Whisper's autodetect is never consulted:
        the caller (DirectionRouter, or a manual pin) has already made that decision
        with more context than the acoustics alone provide.
        """
        code = normalize_lang(lang)
        role = LANG_TO_ROLE[code]

        was_loaded = role in self._models
        t_load = time.perf_counter()
        model = self._ensure(role)
        load_ms = 0.0 if was_loaded else (time.perf_counter() - t_load) * 1000.0

        pcm = resample_to_16k(audio, sr)
        audio_seconds = len(pcm) / float(SAMPLE_RATE)

        t0 = time.perf_counter()
        segments, info = model.transcribe(pcm, **self._decode_kwargs(code))
        # faster-whisper returns a generator; nothing runs until it is consumed.
        seg_list = list(segments)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        meta = self._info[role]
        meta.calls += 1
        meta.total_decode_ms += latency_ms

        return TranscriptionResult(
            text=" ".join(s.text for s in seg_list).strip(),
            lang=code,
            role=role,
            model_path=self.cfg.dir_for_role(role),
            forced_language=WHISPER_LANG[code],
            latency_ms=latency_ms,
            audio_seconds=audio_seconds,
            load_ms=load_ms,
            segments=len(seg_list),
            language_probability=getattr(info, "language_probability", None),
        )

    def transcribe_file(self, path: str | Path, lang: str) -> TranscriptionResult:
        return self.transcribe(load_wav_mono16k(path), lang, SAMPLE_RATE)

    def transcribe_route(self, audio: np.ndarray, decision: Any, sr: int = SAMPLE_RATE):
        """Serve an explicitly pinned `lang_router.RouteDecision`.

        A decision inferred from an ASR transcript cannot safely route that same ASR
        pass. Refuse it instead of presenting post-hoc text LID as acoustic routing.
        """
        if not getattr(decision, "manual", False):
            raise ValueError(
                "ASR source must be selected before decoding; transcript-based "
                "RouteDecision is not a valid ASR route"
            )
        return self.transcribe(audio, decision.source, sr)

    # ---------------- always-warm ----------------

    def prewarm(self, langs: Iterable[str]) -> dict:
        """Load and run one dummy inference for every model the given languages need.

        Pays load + CTranslate2 kernel-selection/JIT cost upfront so the first real
        utterance of a session hits steady state. One second of silence is enough:
        with VAD on, the decoder exits immediately and only the encoder warms, which
        is the expensive half.
        """
        codes = [normalize_lang(x) for x in langs]
        roles: list[str] = []
        for code in codes:
            role = LANG_TO_ROLE[code]
            if role not in roles:
                roles.append(role)

        report: dict[str, dict] = {}
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        for role in roles:
            lang = next(c for c in codes if LANG_TO_ROLE[c] == role)
            t0 = time.perf_counter()
            model = self._ensure(role)
            load_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            try:
                segments, _ = model.transcribe(silent, **self._decode_kwargs(lang))
                for _ in segments:
                    pass
                error = None
            except Exception as exc:  # a failed warm-up is a slow first turn, not a crash
                error = str(exc)
                if self.verbose:
                    print(f"[warn] ASR prewarm failed for {role}: {exc}")
            infer_ms = (time.perf_counter() - t1) * 1000.0
            self._info[role].prewarm_ms = infer_ms
            report[role] = {
                "load_ms": round(load_ms, 1),
                "dummy_infer_ms": round(infer_ms, 1),
                "error": error,
            }
            if self.verbose:
                print(f"[prewarm] {role}: load {load_ms:.0f} ms + dummy {infer_ms:.0f} ms")
        return report

    def prewarm_pair(self, pair: Iterable[str]) -> dict:
        """Prewarm exactly the models a configured competition pair needs.

        vi<->en, vi<->zh and vi<->ko all need both models; a hypothetical vi-only
        session needs one. The saving is real and the caller does not have to know
        the mapping.
        """
        return self.prewarm(tuple(pair))


# ------------------------------------------------------------------ benchmark

def _find_bench_audio() -> list[tuple[str, Path]]:
    """Whatever real audio the repo actually has, tagged with the language to decode as.

    The en/zh/ko files are the repo's own smoke wavs (TTS output, not human speech),
    and the vi files are VIVOS test utterances. This is enough to price latency; it is
    NOT a WER test set and is not used as one.
    """
    here = Path(__file__).resolve().parent
    picks: list[tuple[str, Path]] = []
    for lang, name in (("en", "smoke_R002_en.wav"), ("zh", "smoke_R002_zh.wav"), ("ko", "smoke_R002_ko.wav")):
        p = here / name
        if p.exists():
            picks.append((lang, p))
    vivos = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test" / "waves"
    if vivos.exists():
        for wav in sorted(vivos.rglob("*.wav"))[:3]:
            picks.append(("vi", wav))
    return picks


def run_benchmark(repeats: int = 3) -> dict:
    """Measure load time, per-utterance latency, and the two-model footprint. For real.

    The first decode after a model load is discarded. It is reproducibly 2-3x the
    steady-state number on this hardware (CTranslate2 kernel selection plus a cold
    page cache on a freshly written model.bin), and reporting it as the utterance
    latency would libel the steady-state device. What a conversation device actually
    experiences is the warm number, because `prewarm()` is always called first.
    """
    router = AsrRouter()
    clips = _find_bench_audio()
    if not clips:
        print("[bench] no audio found; nothing to measure")
        return {}

    print("=" * 78)
    print("PREWARM (vi<->en pair: both models)")
    print("=" * 78)
    warm = router.prewarm_pair(("vi", "en"))

    print()
    print("=" * 78)
    print(f"{'lang':<5} {'role':<14} {'audio_s':>8} {'warm_ms':>9} {'rtf':>6}  {'file':<24} text")
    print("=" * 78)
    rows = []
    for lang, path in clips:
        lats = []
        result = None
        for _ in range(max(2, repeats)):
            result = router.transcribe_file(path, lang)
            lats.append(result.latency_ms)
        steady = sorted(lats[1:])  # drop the first decode; see docstring
        warm_ms = steady[len(steady) // 2]
        silent = not result.text
        note = "  <EMPTY: no speech detected>" if silent else ""
        print(
            f"{lang:<5} {result.role:<14} {result.audio_seconds:>8.2f} {warm_ms:>9.0f} "
            f"{(warm_ms/1000)/max(result.audio_seconds,1e-9):>6.2f}  {path.name:<24} "
            f"{result.text[:40]}{note}"
        )
        rows.append(
            {
                "lang": lang,
                "role": result.role,
                "file": str(path),
                "audio_seconds": round(result.audio_seconds, 2),
                "warm_latency_ms": round(warm_ms, 1),
                "first_call_ms": round(lats[0], 1),
                "latencies_ms": [round(x, 1) for x in lats],
                "segments": result.segments,
                "empty": silent,
                "text": result.text,
            }
        )

    fp = router.footprint()
    print()
    print("=" * 72)
    print("FOOTPRINT")
    print("=" * 72)
    for role, info in fp["models"].items():
        print(
            f"  {role:<14} loaded={str(info['loaded']):<5} {info['disk_mb']:>7.1f} MB  "
            f"load={info['load_ms'] and round(info['load_ms']) or '-'} ms  calls={info['calls']}"
        )
    print(f"  RESIDENT TOTAL (both models, on disk): {fp['both_models_disk_mb']:.1f} MB")
    print()
    print("NOTE: latency and footprint are measured; zh/ko ACCURACY is NOT.")
    print("      The repo has no labelled zh/ko reference audio, so no WER is claimed.")
    empties = [r["lang"] for r in rows if r["empty"]]
    if empties:
        print(f"      Clips marked <EMPTY> ({', '.join(empties)}) contain no speech --")
        print("      their latency is a VAD-reject, not a decode. Do not read them as fast.")
    return {"prewarm": warm, "utterances": rows, "footprint": fp}


if __name__ == "__main__":
    import json

    out = run_benchmark()
    if os.environ.get("VOICEAI_BENCH_JSON"):
        Path(os.environ["VOICEAI_BENCH_JSON"]).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
