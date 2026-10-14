"""Kaggle T4 benchmark for the bidirectional cascade and the Round 4-6 changes.

`exp_t4_s2st_inference.py` measured one direction (vi->en) on the pre-Round-4
pipeline. Since then the device gained a second ASR model, terminology control on
MT, direction routing, and a deliberately short first TTS fragment - none of it
measured on a GPU. This script measures all of it in one run.

It is written as decision support, not a victory lap. The headline question is
experiment A: what does the Vietnamese specialist actually buy over one
multilingual model? If the answer is "almost nothing", the device should carry one
ASR instead of two and the whole tone-quantization spine needs rethinking.

Experiments
  A  ASR A/B on Vietnamese: PhoWhisper-small INT8 vs whisper-small multilingual,
     WER and CER on the same VIVOS utterances, same decode settings
  B  Reverse-direction ASR: multilingual model on FLEURS en_us and ko_kr against
     their references. CER leads for Korean - see the note in wer_cer()
  C  All six directions end to end, per-stage latency, RTF, first-audio.
     zh->vi is NOT measurable: there is no Mandarin audio in any of our datasets
  D  Glossary A/B on MT: term-recovery rate with injection on and off. This is a
     term-recovery measurement, NOT a quality score - we have no MT references
  E  Streaming TTS first-audio A/B: fast first fragment on vs off, plus the
     playback headroom that decides whether the trade is safe

Kernel requirements: Internet disabled, GPU T4, and these dataset inputs
  daosyduyminh/voiceai-edge-models-t4
  daosyduyminh/voiceai-t4-asr-multilingual
  daosyduyminh/voiceai-saigonhub-tier-1-speech-vi-zh-ko-en
  daosyduyminh/voiceai-t4-srclib
  daosyduyminh/voiceai-t4-wheels-py312
  daosyduyminh/voiceai-t4-bidir-config
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
KAGGLE_INPUT = Path("/kaggle/input")

DEFAULT_CONFIG = {
    "condition": "bidir-smoke-v1",
    "samples": 2,
    "warmup_samples": 1,
    "asr_ab_samples": 2,
    "reverse_samples": 2,
    "required_gpu_substring": "T4",
    "asr_compute_type": "int8_float16",
    "text_compute_type": "int8_float16",
    "seed": 20260820,
}

# NLLB language codes, mirroring prototype/pipeline_fast.py.
NLLB_CODE = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
DIRECTIONS = [("vi", "en"), ("vi", "zh"), ("vi", "ko"),
              ("en", "vi"), ("ko", "vi"), ("zh", "vi")]


# --------------------------------------------------------------------------
# Kaggle layout. Both helpers are copied verbatim from exp_t4_s2st_inference.py
# because each of their branches was paid for with a failed run: datasets mount
# under /kaggle/input/datasets/<owner>/<slug>/ on the current image, and uploaded
# ZIPs auto-extract into a same-named subfolder.
# --------------------------------------------------------------------------
def dataset_root(slug: str) -> Path:
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    owners = KAGGLE_INPUT / "datasets"
    if owners.is_dir():
        for owner in sorted(owners.iterdir()):
            candidate = owner / slug
            if candidate.is_dir():
                return candidate
    if KAGGLE_INPUT.is_dir():
        matches = sorted((p for p in KAGGLE_INPUT.glob(f"*/*/{slug}") if p.is_dir()),
                         key=lambda p: len(p.parts))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Kaggle dataset not mounted: {slug}")


def unwrap_same_name(path: Path) -> Path:
    while True:
        nested = path / path.name
        if not nested.is_dir():
            return path
        path = nested


def payload_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return unwrap_same_name(direct)
    matches = sorted((p for p in root.rglob(name) if p.is_dir()),
                     key=lambda p: len(p.parts))
    if matches:
        return unwrap_same_name(matches[0])
    raise FileNotFoundError(
        f"Missing payload directory {name!r} under {root} "
        f"(children: {[p.name for p in root.iterdir()] if root.is_dir() else 'n/a'})"
    )


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        path = dataset_root("voiceai-t4-bidir-config") / "benchmark_config.json"
    except FileNotFoundError:
        path = HERE / "bidir_config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    cfg["resolved_config_path"] = str(path)
    return cfg


def select_wheels(wheel_files: list[Path]) -> list[Path]:
    abi_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    chosen: dict[str, Path] = {}
    for path in sorted(wheel_files):
        parts = path.name.split("-")
        if len(parts) < 5:
            continue
        dist, python_tag = parts[0].lower().replace("_", "-"), parts[2]
        pure = python_tag.startswith("py3") or python_tag.startswith("py2.py3")
        if not (pure or abi_tag in python_tag.split(".")):
            continue
        if dist not in chosen or not pure:
            chosen[dist] = path
    return [chosen[key] for key in sorted(chosen)]


def install_offline_wheels(roots: list[Path]) -> None:
    found: list[Path] = []
    for root in roots:
        found.extend(p for p in root.rglob("*.whl") if p.is_file())
    wheels = select_wheels(found)
    if not wheels:
        raise FileNotFoundError(f"No wheels for this interpreter among {[p.name for p in found]}")
    print(f"[wheels] installing {[p.name for p in wheels]}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                    "--no-deps", *map(str, wheels)], check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def normalize_for_scoring(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFC.

    Deliberately does NOT strip Vietnamese diacritics: tone marks are the
    semantic content this project exists to preserve, so folding them away would
    hide exactly the errors we care about.
    """
    text = unicodedata.normalize("NFC", text or "").casefold()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in text]
    return " ".join("".join(kept).split())


def _levenshtein(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    previous = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        current = [i]
        for j, r in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (h != r)))
        previous = current
    return previous[-1]


def wer_cer(reference: str, hypothesis: str) -> dict:
    """Word and character error rate over `normalize_for_scoring` output.

    For Korean, WER over whitespace tokens is a poor metric: spacing is partly
    optional and orthographically inconsistent, so a correct transcript can score
    a high WER purely on segmentation. CER is the number to read for ko, and the
    report says so per language rather than averaging the two into one figure.
    """
    ref_n, hyp_n = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    if not ref_n:
        # An empty reference makes the rate unbounded (deletions over zero words),
        # which would silently inflate any mean it lands in. Mark it unscored.
        return {"wer": None, "cer": None, "ref_words": 0, "ref_chars": 0,
                "ref_norm": "", "hyp_norm": hyp_n, "unscored": "empty reference"}
    ref_w, hyp_w = ref_n.split(), hyp_n.split()
    ref_c, hyp_c = list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", ""))
    return {
        "wer": _levenshtein(ref_w, hyp_w) / max(len(ref_w), 1),
        "cer": _levenshtein(ref_c, hyp_c) / max(len(ref_c), 1),
        "ref_words": len(ref_w), "ref_chars": len(ref_c),
        "ref_norm": ref_n, "hyp_norm": hyp_n,
    }


def pct(values: list[float], q: float) -> float:
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stat_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "reason": "no samples"}
    return {"n": len(values), "mean": statistics.fmean(values),
            "p50": pct(values, 50), "p95": pct(values, 95),
            "min": min(values), "max": max(values)}


class GpuMonitor:
    def __init__(self, index: int = 0):
        self.index = index
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception as exc:
            print(f"[warn] NVML monitor unavailable: {exc}")
        return self

    def _poll(self):
        while not self._stop.is_set():
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            self.samples.append({"used_mib": mem.used / (1024 ** 2),
                                 "total_mib": mem.total / (1024 ** 2),
                                 "gpu_util_pct": util.gpu})
            self._stop.wait(0.1)

    def __exit__(self, *_a):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        return {"available": True,
                "peak_used_mib": max(r["used_mib"] for r in self.samples),
                "total_mib": max(r["total_mib"] for r in self.samples),
                "peak_gpu_util_pct": max(r["gpu_util_pct"] for r in self.samples),
                "n_samples": len(self.samples)}


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------
def load_vivos(speech_root: Path, limit: int) -> list[dict]:
    prompts = {}
    for line in (speech_root / "test" / "prompts.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            prompts[parts[0]] = parts[1]
    out = []
    for wav in sorted((speech_root / "test" / "waves").rglob("*.wav"))[:limit]:
        ref = prompts.get(wav.stem)
        if ref:
            out.append({"id": wav.stem, "path": str(wav), "reference": ref, "lang": "vi"})
    return out


def load_fleurs(parquet_path: Path, lang: str, limit: int) -> list[dict]:
    """Read a HuggingFace FLEURS test parquet.

    The schema is not contractual, so probe it and fail with the real column
    names rather than guessing - a wrong guess here would silently score against
    the wrong field.
    """
    import numpy as np
    import pandas as pd
    import soundfile as sf

    frame = pd.read_parquet(parquet_path)
    columns = list(frame.columns)
    text_col = next((c for c in ("transcription", "raw_transcription", "text", "sentence")
                     if c in columns), None)
    audio_col = next((c for c in ("audio", "audio_bytes", "path") if c in columns), None)
    if text_col is None or audio_col is None:
        raise RuntimeError(
            f"FLEURS parquet {parquet_path} has unexpected columns {columns}; "
            "expected a transcription-like and an audio-like column")

    rows = []
    for i, record in frame.head(limit * 3).iterrows():
        blob = record[audio_col]
        if isinstance(blob, dict):
            raw = blob.get("bytes")
        elif isinstance(blob, (bytes, bytearray)):
            raw = bytes(blob)
        else:
            raw = None
        if raw is None:
            continue
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            n = int(round(len(audio) * 16_000 / sr))
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype("float32")
        rows.append({"id": f"fleurs-{lang}-{i}", "audio": audio,
                     "reference": str(record[text_col]), "lang": lang})
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No decodable audio rows in {parquet_path} (columns {columns})")
    return rows


def read_wav(path: str):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        n = int(round(len(audio) * 16_000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio


def main() -> int:
    cfg = load_config()
    output_root = Path("/kaggle/working")
    if not KAGGLE_INPUT.exists():
        raise RuntimeError("This benchmark is intended for a Kaggle kernel")

    model_root = dataset_root("voiceai-edge-models-t4")
    speech_root = payload_dir(dataset_root("voiceai-saigonhub-tier-1-speech-vi-zh-ko-en"), "vivos")
    speech_ds = dataset_root("voiceai-saigonhub-tier-1-speech-vi-zh-ko-en")
    multi_root = dataset_root("voiceai-t4-asr-multilingual")
    print(f"[paths] model_root={model_root}")
    print(f"[paths] multi_root={multi_root}")
    print(f"[paths] speech_root={speech_root}")

    wheel_roots = [model_root]
    try:
        wheel_roots.insert(0, dataset_root("voiceai-t4-wheels-py312"))
    except FileNotFoundError:
        pass
    install_offline_wheels(wheel_roots)
    sys.path.insert(0, str(HERE))
    try:
        sys.path.insert(0, str(dataset_root("voiceai-t4-srclib")))
    except FileNotFoundError:
        pass

    import ctranslate2
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
    from transformers import BartphoTokenizer, NllbTokenizerFast
    from local_tts_piper import PiperTTS
    from transcript_corrector import CorrectorConfig, TranscriptCorrector
    from glossary_mt import Glossary
    from lang_router import DirectionRouter
    from streaming_tts import StreamingTTSConfig, fragment_text

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not gpu_names or cfg["required_gpu_substring"].lower() not in gpu_names[0].lower():
        (output_root / "voiceai_bidir_failure.json").write_text(
            json.dumps({"status": "failed_gpu_gate", "gpu_names": gpu_names, "config": cfg},
                       indent=2), encoding="utf-8")
        raise RuntimeError(f"T4 required, got {gpu_names}")

    models_dir = payload_dir(model_root, "models_ct2")
    piper_dir = payload_dir(model_root, "piper-models")
    multi_dir = payload_dir(multi_root, "whisper-small-multilingual")
    asr_vi_dir = models_dir / "phowhisper-small-int8"
    mt_dir = models_dir / "nllb-200-distilled-600M-int8"
    corr_dir = models_dir / "vietnamese-correction-v2-int8"
    # The original model bundle carries only the English voice; the other three
    # live in their own dataset. Kaggle inputs are read-only, so merge them into a
    # writable directory of symlinks rather than copying ~340 MB.
    voices_dir = Path("/kaggle/working/piper-voices")
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_sources = [piper_dir]
    try:
        voice_sources.append(payload_dir(dataset_root("voiceai-t4-piper-voices"), "piper-models"))
    except FileNotFoundError:
        print("[warn] extra Piper voices not mounted; non-English targets will fail")
    for source in voice_sources:
        for entry in sorted(source.iterdir()):
            if entry.is_dir() and not (voices_dir / entry.name).exists():
                (voices_dir / entry.name).symlink_to(entry, target_is_directory=True)
    print(f"[voices] {sorted(p.name for p in voices_dir.iterdir())}")
    os.environ["VOICEAI_PIPER_MODELS_DIR"] = str(voices_dir)

    report: dict = {
        "status": "running",
        "config": cfg,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_names": gpu_names,
            "torch": torch.__version__,
            "ctranslate2": ctranslate2.__version__,
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
        },
        "model_sha256": {
            "asr_vi": sha256(asr_vi_dir / "model.bin"),
            "asr_multilingual": sha256(multi_dir / "model.bin"),
            "mt": sha256(mt_dir / "model.bin"),
            "corrector": sha256(corr_dir / "model.bin"),
        },
        "notes": [
            "T4 is a development benchmark, not the competition deployment device.",
            "Piper TTS runs on CPU; ASR, corrector and MT run on CUDA device 0.",
            "Scoring normalization keeps Vietnamese diacritics (see normalize_for_scoring).",
        ],
    }

    load_ms: dict = {}
    with GpuMonitor() as gpu:
        started = time.perf_counter()
        asr_vi = WhisperModel(str(asr_vi_dir), device="cuda", device_index=0,
                              compute_type=cfg["asr_compute_type"], cpu_threads=2, num_workers=1)
        load_ms["asr_vi"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        asr_multi = WhisperModel(str(multi_dir), device="cuda", device_index=0,
                                 compute_type=cfg["asr_compute_type"], cpu_threads=2, num_workers=1)
        load_ms["asr_multilingual"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        corr_translator = ctranslate2.Translator(
            str(corr_dir), device="cuda", device_index=0,
            compute_type=cfg["text_compute_type"], inter_threads=1)
        corr_tok = BartphoTokenizer.from_pretrained(str(corr_dir), local_files_only=True)

        def correct_generate(text: str) -> str:
            ids = corr_tok(text, add_special_tokens=True)["input_ids"]
            tokens = corr_tok.convert_ids_to_tokens(ids)
            result = corr_translator.translate_batch(
                [tokens], beam_size=1,
                max_decoding_length=min(256, max(len(tokens) + 16, 24)),
                repetition_penalty=1.05, no_repeat_ngram_size=4)[0]
            return corr_tok.decode(corr_tok.convert_tokens_to_ids(result.hypotheses[0]),
                                   skip_special_tokens=True)

        corrector = TranscriptCorrector(
            CorrectorConfig(dynamic_int8=False, ct2_model=""), generator=correct_generate)
        load_ms["corrector"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        mt = ctranslate2.Translator(str(mt_dir), device="cuda", device_index=0,
                                    compute_type=cfg["text_compute_type"], inter_threads=1)
        mt_tok = NllbTokenizerFast.from_pretrained(str(mt_dir), local_files_only=True)
        load_ms["mt"] = (time.perf_counter() - started) * 1000

        # cache_cap=4 so cycling through targets does not evict and re-load a
        # voice mid-measurement. The device ships cap=2 because exactly two
        # languages are live per session; here we are timing synthesis, not
        # eviction, and a cold voice load would otherwise be charged to tts_ms.
        started = time.perf_counter()
        # Pass the thread count through the ENV, never the constructor: `PiperTTS`
        # sets `_thread_override = num_threads is not None`, and `_threads_for`
        # returns early on that flag — so a constructor argument silently disables
        # the per-language budget this run exists to measure. en/vi/ko stay at 2 as
        # a control; only Mandarin picks up its own budget.
        os.environ.setdefault("VOICEAI_TTS_THREADS", "2")
        tts = PiperTTS(auto_download=False, cache_cap=4)
        voice_load_ms = {}
        for lang in ("en", "vi", "zh", "ko"):
            t0 = time.perf_counter()
            tts.synthesize("Ready.", lang)
            voice_load_ms[lang] = (time.perf_counter() - t0) * 1000
        load_ms["tts_all_voices"] = (time.perf_counter() - started) * 1000
        report["tts_voice_load_ms"] = voice_load_ms
        report["cpu_count"] = os.cpu_count()
        report["tts_threads_per_lang"] = {
            lang: tts._threads_for(tts._voice_for(lang)) for lang in ("en", "vi", "zh", "ko")
        }
        report["model_load_ms"] = load_ms

        glossary = Glossary()

        def transcribe(model, audio, lang: str) -> tuple[str, float]:
            t0 = time.perf_counter()
            segments, _ = model.transcribe(audio, language=lang, beam_size=1,
                                           vad_filter=True, condition_on_previous_text=False)
            text = " ".join(seg.text for seg in segments).strip()
            return text, (time.perf_counter() - t0) * 1000

        def translate(text: str, source: str, target: str, use_glossary: bool) -> tuple[str, float, dict]:
            meta = {"injected": False, "exact_hit": False, "terms": []}
            if use_glossary:
                prepared = glossary.inject(text, source)
                meta["terms"] = [t.en for t in prepared.terms]
                if prepared.exact_hit is not None:
                    meta["exact_hit"] = True
                    return prepared.exact_hit.get(target), 0.0, meta
                meta["injected"] = prepared.changed
                text = prepared.text
            t0 = time.perf_counter()
            mt_tok.src_lang = NLLB_CODE[source]
            tokens = mt_tok.convert_ids_to_tokens(mt_tok.encode(text))
            result = mt.translate_batch([tokens], target_prefix=[[NLLB_CODE[target]]],
                                        beam_size=1,
                                        max_decoding_length=min(256, max(16, len(tokens) * 2 + 4)))
            out = mt_tok.decode(mt_tok.convert_tokens_to_ids(result[0].hypotheses[0][1:]),
                                skip_special_tokens=True).strip()
            return out, (time.perf_counter() - t0) * 1000, meta

        # ---------------- A: ASR A/B on Vietnamese ----------------
        vivos = load_vivos(speech_root, int(cfg["asr_ab_samples"]) + int(cfg["warmup_samples"]))
        warm_n = int(cfg["warmup_samples"])
        for item in vivos[:warm_n]:
            audio = read_wav(item["path"])
            transcribe(asr_vi, audio, "vi")
            transcribe(asr_multi, audio, "vi")

        asr_ab_rows = []
        for item in vivos[warm_n:]:
            audio = read_wav(item["path"])
            dur = len(audio) / 16_000
            vi_text, vi_ms = transcribe(asr_vi, audio, "vi")
            mu_text, mu_ms = transcribe(asr_multi, audio, "vi")
            asr_ab_rows.append({
                "id": item["id"], "duration_s": dur, "reference": item["reference"],
                "phowhisper": {"text": vi_text, "latency_ms": vi_ms, **wer_cer(item["reference"], vi_text)},
                "multilingual": {"text": mu_text, "latency_ms": mu_ms, **wer_cer(item["reference"], mu_text)},
            })

        def agg(rows, model_key):
            wers = [r[model_key]["wer"] for r in rows if r[model_key]["wer"] is not None]
            cers = [r[model_key]["cer"] for r in rows if r[model_key]["cer"] is not None]
            return {
                "wer_mean": statistics.fmean(wers) if wers else None,
                "cer_mean": statistics.fmean(cers) if cers else None,
                "n_scored": len(wers),
                "latency_ms": stat_block([r[model_key]["latency_ms"] for r in rows]),
            }

        report["experiment_a_asr_ab_vietnamese"] = {
            "corpus": "VIVOS test",
            "n": len(asr_ab_rows),
            "phowhisper_small_int8": agg(asr_ab_rows, "phowhisper"),
            "whisper_small_multilingual": agg(asr_ab_rows, "multilingual"),
            "rows": asr_ab_rows,
            "interpretation": ("Lower WER/CER is better. This prices the Vietnamese "
                               "specialist against carrying one multilingual model."),
        }

        # ---------------- B: reverse-direction ASR ----------------
        reverse = {}
        # Mandarin lives in its own dataset: the tier-1 speech bundle shipped
        # vi/en/ko only, which is why every earlier zh figure was a null.
        fleurs_specs = [("en", "en_us"), ("ko", "ko_kr"), ("zh", "cmn_hans_cn")]
        fleurs_roots = {"en": speech_ds, "ko": speech_ds}
        try:
            fleurs_roots["zh"] = dataset_root("voiceai-t4-fleurs-zh")
        except FileNotFoundError:
            pass
        for lang, folder in fleurs_specs:
            try:
                root = fleurs_roots.get(lang)
                if root is None:
                    raise FileNotFoundError("FLEURS Mandarin dataset not mounted")
                base = payload_dir(root, folder)
                parquet = next(iter(sorted(base.glob("test-*.parquet"))), None)
                if parquet is None:
                    raise FileNotFoundError(f"no test parquet under {base}")
                rows = load_fleurs(parquet, lang, int(cfg["reverse_samples"]))
            except Exception as exc:
                reverse[lang] = {"measured": False, "reason": f"{type(exc).__name__}: {exc}"}
                continue
            scored = []
            for item in rows:
                text, ms = transcribe(asr_multi, item["audio"], lang)
                scored.append({"id": item["id"], "duration_s": len(item["audio"]) / 16_000,
                               "reference": item["reference"], "text": text,
                               "latency_ms": ms, **wer_cer(item["reference"], text)})
            reverse[lang] = {
                "measured": True, "corpus": f"FLEURS {folder} test", "n": len(scored),
                "wer_mean": statistics.fmean([r["wer"] for r in scored if r["wer"] is not None]),
                "cer_mean": statistics.fmean([r["cer"] for r in scored if r["cer"] is not None]),
                "n_scored": sum(r["wer"] is not None for r in scored),
                # Korean spacing is partly optional and FLEURS Mandarin ships a
                # character-spaced reference, so whitespace WER is meaningless
                # for both. CER is the metric that means anything here.
                "primary_metric": "cer" if lang in ("ko", "zh") else "wer",
                "latency_ms": stat_block([r["latency_ms"] for r in scored]),
                "rows": scored,
            }
        report["experiment_b_reverse_asr"] = reverse

        # ---------------- C: six directions end to end ----------------
        source_audio = {"vi": [(i["id"], read_wav(i["path"])) for i in vivos[warm_n:]]}
        for lang in ("en", "ko", "zh"):
            block = reverse.get(lang, {})
            if block.get("measured"):
                base = payload_dir(fleurs_roots[lang], dict(fleurs_specs)[lang])
                parquet = next(iter(sorted(base.glob("test-*.parquet"))))
                source_audio[lang] = [(r["id"], r["audio"])
                                      for r in load_fleurs(parquet, lang, int(cfg["samples"]))]

        direction_rows = []
        direction_status = {}
        for source, target in DIRECTIONS:
            clips = source_audio.get(source)
            if not clips:
                direction_status[f"{source}->{target}"] = {
                    "measured": False,
                    "reason": reverse.get(source, {}).get("reason", f"no {source} audio available")}
                continue
            model = asr_vi if source == "vi" else asr_multi
            rows = []
            for clip_id, audio in clips[: int(cfg["samples"])]:
                dur = len(audio) / 16_000
                t_start = time.perf_counter()
                raw, asr_ms = transcribe(model, audio, source)
                corr_ms, corrected, reason = 0.0, raw, "skipped_non_vi"
                if source == "vi":
                    t0 = time.perf_counter()
                    result = corrector.correct(raw)
                    corr_ms = (time.perf_counter() - t0) * 1000
                    corrected, reason = result.text, result.reason
                translated, mt_ms, meta = translate(corrected, source, target, use_glossary=True)
                t0 = time.perf_counter()
                wav, sr = tts.synthesize(translated, target)
                tts_ms = (time.perf_counter() - t0) * 1000
                total_ms = (time.perf_counter() - t_start) * 1000
                rows.append({
                    "direction": f"{source}->{target}", "id": clip_id, "duration_s": dur,
                    "raw": raw, "corrected": corrected, "correction_reason": reason,
                    "translation": translated, "glossary": meta,
                    "asr_ms": asr_ms, "corrector_ms": corr_ms, "mt_ms": mt_ms,
                    "tts_ms": tts_ms, "total_ms": total_ms,
                    "output_audio_s": len(wav) / sr if sr else 0.0,
                    "rtf": total_ms / 1000 / max(dur, 1e-6),
                })
            direction_rows.extend(rows)
            direction_status[f"{source}->{target}"] = {
                "measured": True, "n": len(rows),
                "total_ms": stat_block([r["total_ms"] for r in rows]),
                "rtf": stat_block([r["rtf"] for r in rows]),
                "asr_ms": stat_block([r["asr_ms"] for r in rows]),
                "mt_ms": stat_block([r["mt_ms"] for r in rows]),
                "tts_ms": stat_block([r["tts_ms"] for r in rows]),
                "rtf_under_1": sum(r["rtf"] < 1.0 for r in rows) / len(rows),
                "under_2s": sum(r["total_ms"] < 2000 for r in rows) / len(rows),
            }
        report["experiment_c_directions"] = direction_status

        # ---------------- D: glossary A/B ----------------
        # Probe set MUST NOT be drawn from the glossary. v2 built it that way and the
        # consequence only surfaced later: growing the table 61 -> 93 terms left
        # find_terms over the fixed sentences returning the identical 34 terms, so a
        # coverage measurement could not detect coverage. These sentences are
        # authored independently and exercise terms across the whole table.
        probe_path = None
        for candidate in (Path(sys.path[0]) / "glossary_probe_sentences.json",
                          HERE.parent / "prototype" / "glossary_probe_sentences.json"):
            if candidate.is_file():
                probe_path = candidate
                break
        if probe_path is not None:
            probes = json.loads(probe_path.read_text(encoding="utf-8"))["sentences"]
            probe_source = str(probe_path)
        else:
            probes = ["Xe nâng đang lùi, xin tránh ra phía sau.",
                      "Máy nén khí ở khu vực B đang quá nhiệt, hãy dừng dây chuyền ngay.",
                      "Anh phải đeo găng tay và kính bảo hộ trước khi vào xưởng."]
            probe_source = "inline fallback (held-out file not mounted)"
        # A handful of bare terms stay in, purely to keep the exact-hit path exercised.
        probes = probes + [t.vi for t in glossary.terms][:6]
        glossary_rows = []
        for target in ("en", "zh", "ko"):
            for probe in probes:
                off_text, off_ms, _ = translate(probe, "vi", target, use_glossary=False)
                on_text, on_ms, meta = translate(probe, "vi", target, use_glossary=True)
                expected = [t for t in glossary.find_terms(probe, "vi")]
                def has(term_text: str, out: str) -> bool:
                    return normalize_for_scoring(term_text) in normalize_for_scoring(out)
                glossary_rows.append({
                    "target": target, "source_text": probe,
                    "off": off_text, "on": on_text, "changed": off_text != on_text,
                    "exact_hit": meta["exact_hit"],
                    "expected_terms": [t.get(target) for t in expected],
                    "recovered_off": sum(has(t.get(target), off_text) for t in expected),
                    "recovered_on": sum(has(t.get(target), on_text) for t in expected),
                    "n_expected": len(expected),
                    "off_ms": off_ms, "on_ms": on_ms,
                })
        def recovery(rows):
            expected = sum(r["n_expected"] for r in rows)
            if not expected:
                return {"n_terms": 0, "reason": "no glossary term in these probes"}
            return {"n_probes": len(rows), "n_terms": expected,
                    "off": sum(r["recovered_off"] for r in rows) / expected,
                    "on": sum(r["recovered_on"] for r in rows) / expected}

        with_terms = [r for r in glossary_rows if r["n_expected"] > 0]
        total_expected = sum(r["n_expected"] for r in with_terms)
        # An exact whole-utterance hit bypasses the decoder entirely, so counting
        # it as "recovered" measures the lookup table, not the injection. The
        # sentence split is the honest number.
        exact_rows = [r for r in with_terms if r["exact_hit"]]
        sentence_rows = [r for r in with_terms if not r["exact_hit"]]
        report["experiment_d_glossary"] = {
            "probe_source": probe_source,
            "n_glossary_terms": len(glossary.terms),
            "n_probes": len(glossary_rows),
            "changed_rate": sum(r["changed"] for r in glossary_rows) / max(len(glossary_rows), 1),
            "exact_hit_rate": sum(r["exact_hit"] for r in glossary_rows) / max(len(glossary_rows), 1),
            "term_recovery_all": recovery(with_terms),
            "term_recovery_exact_hits": recovery(exact_rows),
            "term_recovery_translated_sentences": recovery(sentence_rows),
            "term_recovery_off": sum(r["recovered_off"] for r in with_terms) / max(total_expected, 1),
            "term_recovery_on": sum(r["recovered_on"] for r in with_terms) / max(total_expected, 1),
            "n_terms_expected": total_expected,
            "caveat": ("Term recovery counts the glossary's preferred target surface "
                       "form appearing in the output. It is NOT an adequacy score: we "
                       "have no MT references for these sentences, and a correct "
                       "synonym counts as a miss."),
            "rows": glossary_rows,
        }

        # ---------------- E: streaming TTS first audio ----------------
        tts_probe = ("Nhiệt độ trong nhà máy hiện tại là bốn mươi lăm độ C, vượt ngưỡng "
                     "an toàn, hãy dừng dây chuyền ngay. Cảm ơn.")
        tts_probe_en, _, _ = translate(tts_probe, "vi", "en", use_glossary=True)
        tts_ab = {}
        for label, config in (("fast_first", StreamingTTSConfig()),
                              ("whole_sentence", StreamingTTSConfig(min_first_fragment_chars=10_000))):
            frags = fragment_text(tts_probe_en, config)
            tts.synthesize(frags[0], "en")
            first_ms, total_ms, durations, synth_ms = None, None, [], []
            t_start = time.perf_counter()
            for i, frag in enumerate(frags):
                t0 = time.perf_counter()
                wav, sr = tts.synthesize(frag, "en")
                ms = (time.perf_counter() - t0) * 1000
                synth_ms.append(ms)
                durations.append(len(wav) / sr if sr else 0.0)
                if i == 0:
                    first_ms = (time.perf_counter() - t_start) * 1000
            total_ms = (time.perf_counter() - t_start) * 1000
            headroom = (durations[0] * 1000 - synth_ms[1]) if len(frags) > 1 else None
            tts_ab[label] = {
                "n_fragments": len(frags), "first_audio_ms": first_ms,
                "total_synth_ms": total_ms, "fragment_synth_ms": synth_ms,
                "fragment_audio_s": durations,
                "playback_headroom_ms": headroom,
                "first_fragment": frags[0],
            }
        tts_ab["note"] = ("playback_headroom_ms = fragment 0 audio duration minus "
                          "fragment 1 synthesis time. Negative means the player would "
                          "underrun and the fast first fragment is a bad trade.")
        report["experiment_e_streaming_tts"] = tts_ab

        report["gpu_usage"] = gpu.summary()

    report["status"] = "complete"
    report["direction_rows"] = direction_rows
    out_name = f"voiceai_{cfg['condition']}.json"
    (output_root / out_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / f"voiceai_{cfg['condition']}_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in direction_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({k: report[k] for k in
                      ("experiment_a_asr_ab_vietnamese", "experiment_c_directions",
                       "experiment_d_glossary", "gpu_usage")
                      if k in report}, ensure_ascii=False, indent=2, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
