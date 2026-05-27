"""
Optimized cascade S2ST for CPU local — full-stack INT8 inference.

Drop-in replacement for pipeline.py with stage-by-stage optimization:

  Stage 1 — ASR:  faster-whisper (CTranslate2 backend) INT8
                  + beam=1 greedy + built-in silero-VAD endpoint trimming
                  -> 4-8x faster than HF WhisperForConditionalGeneration

  Stage 2 — MT:   CTranslate2 Translator INT8 for NLLB-200-distilled-600M
                  + beam=1 greedy + dynamic max_decoding_length
                  -> 5-10x faster than HF AutoModelForSeq2SeqLM

  Stage 3 — TTS:  torch.quantization.quantize_dynamic INT8 on MMS-TTS VITS
                  Linear layers
                  -> 1.3-2x faster than FP32 (Conv1d-dominant model, modest win)

Optional knobs (env):
  VOICEAI_GREEDY=1            (default) -- beam=1 everywhere
  VOICEAI_LOAD_PARALLEL=1     (default) -- load TTS in a thread while MT decodes
  VOICEAI_CT2_INTRA_THREADS=4 -- CTranslate2 intra-op threads (auto = #cores)

Cache layout (set up by `python -m ctranslate2.converters.transformers ...`):
  data/raw/models_ct2/phowhisper-small-int8/           CT2 Whisper format
  data/raw/models_ct2/nllb-200-distilled-600M-int8/    CT2 NLLB format

Run:
  python prototype/pipeline_fast.py \
      --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \
      --target en --out prototype/out_fast.wav

Compare against the FP32 baseline pipeline.py with prototype/bench_compare.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Redirect HF caches BEFORE imports (Windows-disk-full guard, same as pipeline.py).
PROJECT = Path(__file__).resolve().parents[1]
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
(HF_CACHE / "xet").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME",            str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",       str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE",       str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

# Force UTF-8 stdout (Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try: _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                          # noqa: E402
import soundfile as sf                                       # noqa: E402


# Per-target NLLB language codes (BCP-47 + script).
NLLB_CODE = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}

# Per-target MMS-TTS checkpoints (same as pipeline.py).
TTS_REPO = {
    "vi": "facebook/mms-tts-vie",
    "en": "facebook/mms-tts-eng",
    "zh": None,
    "ko": "facebook/mms-tts-kor",
}

# Languages whose MMS-TTS tokenizer needs romanization.
_ROMANIZE = {"ko"}


# --- Sentence splitter used by streaming MT to dispatch sentences as they arrive ---
import re as _re                                                  # noqa: E402

_SENT_RE = _re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$", flags=_re.UNICODE)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter that keeps trailing punctuation. Vietnamese commonly
    omits final punctuation, so we also break on ", " when the chunk before is
    > 8 chars (heuristic: avoids splitting "anh, hỏi" but allows "tôi đi học, sau đó về nhà")."""
    if not text or not text.strip():
        return []
    parts = _SENT_RE.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # secondary split on commas if chunk is long
        if len(p) > 60 and "," in p:
            for sub in p.split(","):
                sub = sub.strip()
                if sub:
                    out.append(sub)
        else:
            out.append(p)
    return out or [text.strip()]


CT2_ROOT = PROJECT / "data" / "raw" / "models_ct2"


@dataclass
class FastPipelineConfig:
    asr_ct2_dir:    str = str(CT2_ROOT / "phowhisper-small-int8")
    mt_ct2_dir:     str = str(CT2_ROOT / "nllb-200-distilled-600M-int8")
    mt_tokenizer_hf: str = "facebook/nllb-200-distilled-600M"
    asr_lang:       str = "vi"

    # INT8 by default; ct2 also supports int8_float16, float16, etc.
    asr_compute_type: str = "int8"
    mt_compute_type:  str = "int8"

    # Decoding
    asr_beam:       int = 1               # greedy ASR
    mt_beam:        int = 1               # greedy MT
    asr_vad_filter: bool = True           # use faster-whisper's built-in silero-VAD trim
    asr_vad_min_silence_ms: int = 500
    max_decoding_length: int = 256

    # Threading
    ct2_intra_threads: int = int(os.environ.get("VOICEAI_CT2_INTRA_THREADS", 0)) or 0   # 0 = ct2 picks
    ct2_inter_threads: int = 1

    # Parallel load (load TTS in a worker thread while MT first-runs)
    load_parallel: bool = bool(int(os.environ.get("VOICEAI_LOAD_PARALLEL", "1")))

    # Tonal-perceptual utterance cache — see utterance_cache.py.
    # On cache hit, end-to-end latency drops from ~3 s to <100 ms (the fingerprint compute itself).
    cache_enabled:     bool  = bool(int(os.environ.get("VOICEAI_CACHE", "1")))
    cache_max_entries: int   = int(os.environ.get("VOICEAI_CACHE_MAX", "128"))
    cache_cosine_hit:  float = float(os.environ.get("VOICEAI_CACHE_COSINE", "0.95"))

    # Translation memory — text-level fuzzy cache (see translation_memory.py). Hits
    # when a *different* speaker says a similar Vi phrase. Reuses Vi+target text;
    # only re-runs TTS (or even reuses TTS audio when text fingerprint cos == 1).
    tm_enabled:     bool  = bool(int(os.environ.get("VOICEAI_TM", "1")))
    tm_max_entries: int   = int(os.environ.get("VOICEAI_TM_MAX", "256"))
    tm_cosine_hit:  float = float(os.environ.get("VOICEAI_TM_COSINE", "0.80"))

    # Tone-energy VAD — pre-filter to skip cascade on noise / silence / hum.
    # Complementary to silero-VAD in faster-whisper (which fires inside the model).
    tone_vad_enabled: bool = bool(int(os.environ.get("VOICEAI_TONE_VAD", "1")))


class FastCascadeS2ST:
    """Same surface area as CascadeS2ST but with INT8 backends + utterance cache."""

    def __init__(self, cfg: Optional[FastPipelineConfig] = None):
        self.cfg = cfg or FastPipelineConfig()
        self._asr = None
        self._mt = None
        self._mt_tok = None
        self._tts_cache: dict[str, tuple] = {}
        self._tts_thread: Optional[threading.Thread] = None
        # Tier 1 — Audio-level perceptual cache.
        self._utt_cache = None
        if self.cfg.cache_enabled:
            try:
                from utterance_cache import UtteranceCache
                self._utt_cache = UtteranceCache(
                    max_entries=self.cfg.cache_max_entries,
                    cosine_hit=self.cfg.cache_cosine_hit,
                )
                print(f"[init] T1 audio cache enabled  (cos>={self.cfg.cache_cosine_hit}, max={self.cfg.cache_max_entries})")
            except ImportError as e:
                print(f"[warn] utterance_cache module not importable ({e}); T1 disabled")
                self._utt_cache = None

        # Tier 2 — Text-level translation memory.
        self._tm = None
        if self.cfg.tm_enabled:
            try:
                from translation_memory import TranslationMemory
                self._tm = TranslationMemory(
                    max_entries=self.cfg.tm_max_entries,
                    cosine_hit=self.cfg.tm_cosine_hit,
                )
                print(f"[init] T2 translation memory enabled  (cos>={self.cfg.tm_cosine_hit}, max={self.cfg.tm_max_entries})")
            except ImportError as e:
                print(f"[warn] translation_memory module not importable ({e}); T2 disabled")
                self._tm = None

        # Pre-filter — tonal-language VAD (rejects noise/silence/hum before cascade).
        self._tone_vad = None
        if self.cfg.tone_vad_enabled:
            try:
                from tone_vad import ToneVAD
                self._tone_vad = ToneVAD()
                print(f"[init] tone-energy VAD enabled")
            except ImportError as e:
                print(f"[warn] tone_vad module not importable ({e}); VAD disabled")
                self._tone_vad = None

    # ---------------- ASR (faster-whisper) ----------------
    def _ensure_asr(self) -> None:
        if self._asr is not None:
            return
        from faster_whisper import WhisperModel
        if not Path(self.cfg.asr_ct2_dir).exists():
            raise FileNotFoundError(
                f"CT2 ASR model not found at {self.cfg.asr_ct2_dir}. "
                f"Run: python -m ctranslate2.converters.transformers "
                f"--model vinai/PhoWhisper-small --output_dir {self.cfg.asr_ct2_dir} --quantization int8"
            )
        print(f"[load] ASR (faster-whisper): {self.cfg.asr_ct2_dir}  compute={self.cfg.asr_compute_type}")
        self._asr = WhisperModel(
            self.cfg.asr_ct2_dir,
            device="cpu",
            compute_type=self.cfg.asr_compute_type,
            cpu_threads=self.cfg.ct2_intra_threads,
            num_workers=self.cfg.ct2_inter_threads,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        self._ensure_asr()
        if sr != 16000:
            target_len = int(round(len(audio) * 16000 / sr))
            audio = np.interp(np.linspace(0, len(audio), target_len, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        segments, _info = self._asr.transcribe(
            audio.astype(np.float32),
            language=self.cfg.asr_lang,
            beam_size=self.cfg.asr_beam,
            vad_filter=self.cfg.asr_vad_filter,
            vad_parameters=dict(min_silence_duration_ms=self.cfg.asr_vad_min_silence_ms),
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text for s in segments).strip()

    # ---------------- MT (CTranslate2 NLLB) ----------------
    def _ensure_mt(self) -> None:
        if self._mt is not None:
            return
        import ctranslate2
        from transformers import NllbTokenizerFast
        if not Path(self.cfg.mt_ct2_dir).exists():
            raise FileNotFoundError(
                f"CT2 MT model not found at {self.cfg.mt_ct2_dir}. "
                f"Run: python -m ctranslate2.converters.transformers "
                f"--model {self.cfg.mt_tokenizer_hf} --output_dir {self.cfg.mt_ct2_dir} --quantization int8"
            )
        print(f"[load] MT  (CTranslate2): {self.cfg.mt_ct2_dir}  compute={self.cfg.mt_compute_type}")
        self._mt = ctranslate2.Translator(
            self.cfg.mt_ct2_dir,
            device="cpu",
            compute_type=self.cfg.mt_compute_type,
            intra_threads=self.cfg.ct2_intra_threads,
            inter_threads=self.cfg.ct2_inter_threads,
        )
        # Tokenizer still from HF (CT2 doesn't ship one).
        self._mt_tok = NllbTokenizerFast.from_pretrained(self.cfg.mt_tokenizer_hf)

    def translate(self, text: str, target: str, source: str = "vi") -> str:
        if not text.strip():
            return ""
        self._ensure_mt()
        src_code = NLLB_CODE[source]
        tgt_code = NLLB_CODE[target]
        self._mt_tok.src_lang = src_code
        # CT2's translator works on string tokens (not IDs). Pad/append correctly.
        source_tokens = self._mt_tok.convert_ids_to_tokens(self._mt_tok.encode(text))
        target_prefix = [tgt_code]
        # Dynamic max length: ~1.5x source length is a sensible cap for many-to-many MT.
        max_dec = min(self.cfg.max_decoding_length, max(16, int(len(source_tokens) * 2) + 4))
        results = self._mt.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
            beam_size=self.cfg.mt_beam,
            max_decoding_length=max_dec,
            return_scores=False,
        )
        # results[0].hypotheses[0] = [tgt_lang_token, ...tokens, </s>]
        out_tokens = results[0].hypotheses[0][1:]   # drop the language-prefix token
        return self._mt_tok.decode(
            self._mt_tok.convert_tokens_to_ids(out_tokens),
            skip_special_tokens=True,
        ).strip()

    # ---------------- TTS (MMS-TTS + dynamic INT8) ----------------
    def _build_tts(self, target: str):
        import torch
        from transformers import AutoTokenizer, VitsModel
        repo = TTS_REPO[target]
        print(f"[load] TTS (INT8 dynamic quant): {repo}")
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo).eval()
        # Dynamic INT8 over Linear (Conv1d is harder to dynamically quantize and is the bulk
        # of VITS — modest speedup expected, but every ms counts).
        try:
            model_q = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"[warn] dynamic INT8 quantization failed ({e}); using FP32 TTS")
            model_q = model
        return tok, model_q

    def _ensure_tts(self, target: str):
        if target in self._tts_cache:
            return self._tts_cache[target]
        # If a thread was pre-loading this target, join it.
        if self._tts_thread is not None and self._tts_thread.is_alive():
            self._tts_thread.join()
            if target in self._tts_cache:
                return self._tts_cache[target]
        self._tts_cache[target] = self._build_tts(target)
        return self._tts_cache[target]

    def _bg_preload_tts(self, target: str) -> None:
        if not self.cfg.load_parallel or target in self._tts_cache:
            return
        if TTS_REPO[target] is None:
            return
        def _worker():
            self._tts_cache[target] = self._build_tts(target)
        self._tts_thread = threading.Thread(target=_worker, daemon=True)
        self._tts_thread.start()

    def synthesize(self, text: str, target: str) -> tuple[np.ndarray, int]:
        if TTS_REPO[target] is None:
            print(f"[warn] no MMS-TTS for '{target}' yet — emitting 1s silence")
            return np.zeros(16000, dtype=np.float32), 16000
        import torch
        tok, model = self._ensure_tts(target)
        if not text.strip():
            return np.zeros(0, dtype=np.float32), getattr(model.config, "sampling_rate", 16000)
        tts_text = text
        if target in _ROMANIZE:
            try:
                from unidecode import unidecode
                tts_text = unidecode(text).strip()
            except ImportError:
                pass
        inputs = tok(tts_text, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            print(f"[warn] TTS produced 0 tokens for {target} input {text!r}")
            return np.zeros(16000, dtype=np.float32), getattr(model.config, "sampling_rate", 16000)
        with torch.inference_mode():
            wav = model(**inputs).waveform
        return wav.squeeze().cpu().numpy().astype(np.float32), model.config.sampling_rate

    # ---------------- Persistence helpers ----------------
    def save_snapshot(self, name: str = "default") -> dict:
        from persistent_cache import snapshot_pipeline
        return snapshot_pipeline(self, name=name)

    def load_snapshot(self, name: str = "default") -> dict:
        from persistent_cache import restore_pipeline
        return restore_pipeline(self, name=name)

    # ---------------- Streaming end-to-end ----------------
    def run_streaming(self, audio: np.ndarray, sr: int, target: str):
        """Generator that yields progressive results — perceived latency <1s
        because user sees Vi transcript appear word-by-word, then target text
        appear sentence-by-sentence, then audio playback begin before
        synthesis is fully done.

        Yields dicts with keys: stage, vi_text, tgt_text, tgt_audio, latency.
        Stages emitted in order: cache_check -> asr_partial* -> asr_done ->
        mt_partial* -> mt_done -> tts_done -> done.
        """
        import time as _t

        t0 = _t.perf_counter()

        # ----- Pre-filter: tone-energy VAD (reject noise/silence/hum) -----
        if self._tone_vad is not None:
            vad_result = self._tone_vad.is_likely_speech(audio, sr)
            if not vad_result.is_speech:
                yield {
                    "stage": "vad_reject",
                    "vi_text":  "(no speech detected)",
                    "tgt_text": f"(rejected by tone-VAD: {vad_result.reason})",
                    "tgt_audio": np.zeros(16000, dtype=np.float32),
                    "sr":       16000,
                    "latency":  {"vad_ms": round((_t.perf_counter() - t0) * 1000, 2),
                                 "cache_ms": 0, "asr_ms": 0, "mt_ms": 0, "tts_ms": 0,
                                 "total_ms": round((_t.perf_counter() - t0) * 1000, 2)},
                    "cache":    {"hit": False, "cosine": None, "tm_hit": False,
                                 "stats": self._utt_cache.summary() if self._utt_cache else None,
                                 "tm_stats": self._tm.summary() if self._tm else None,
                                 "vad_rejected": True,
                                 "vad_confidence": round(vad_result.confidence, 3)},
                }
                return

        # ----- Cache pre-check (same as run()) -----
        cache_ms = 0.0
        fp = None
        if self._utt_cache is not None:
            tc0 = _t.perf_counter()
            fp = self._utt_cache.fingerprint(audio, sr)
            entry = self._utt_cache.lookup(fp, target)
            cache_ms = (_t.perf_counter() - tc0) * 1000
            if entry is not None:
                res = entry["result"]
                yield {
                    "stage": "cache_hit",
                    "vi_text": res["vi"],
                    "tgt_text": res["tgt"],
                    "tgt_audio": res["audio"][1],
                    "sr": res["audio"][0],
                    "latency": {"cache_ms": round(cache_ms, 2),
                                "asr_ms": 0, "mt_ms": 0, "tts_ms": 0,
                                "total_ms": round(cache_ms, 2)},
                    "cache": {"hit": True, "cosine": round(entry["cosine"], 4),
                              "stats": self._utt_cache.summary()},
                }
                return

        # ----- ASR streaming (faster-whisper emits segments as it decodes) -----
        self._ensure_asr()
        if sr != 16000:
            target_len = int(round(len(audio) * 16000 / sr))
            audio = np.interp(np.linspace(0, len(audio), target_len, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
        t_asr_start = _t.perf_counter()
        segments_iter, _info = self._asr.transcribe(
            audio.astype(np.float32),
            language=self.cfg.asr_lang,
            beam_size=self.cfg.asr_beam,
            vad_filter=self.cfg.asr_vad_filter,
            vad_parameters=dict(min_silence_duration_ms=self.cfg.asr_vad_min_silence_ms),
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        vi_text_acc = ""
        # As each segment finalizes, emit incremental Vi text
        for seg in segments_iter:
            vi_text_acc = (vi_text_acc + " " + seg.text).strip() if vi_text_acc else seg.text.strip()
            asr_ms_so_far = (_t.perf_counter() - t_asr_start) * 1000
            yield {
                "stage": "asr_partial",
                "vi_text": vi_text_acc,
                "tgt_text": "",
                "tgt_audio": None,
                "sr": 16000,
                "latency": {"cache_ms": round(cache_ms, 2),
                            "asr_ms": round(asr_ms_so_far, 1),
                            "mt_ms": 0, "tts_ms": 0,
                            "total_ms": round((cache_ms + asr_ms_so_far), 1)},
                "cache": {"hit": False, "cosine": None,
                          "stats": self._utt_cache.summary() if self._utt_cache else None},
            }
        asr_ms = (_t.perf_counter() - t_asr_start) * 1000

        # ----- Tier 2 cache check: text-level translation memory (post-ASR) -----
        # Looks for previously seen Vi phrases (cos>=0.80). Hit -> reuse target text +
        # audio if available; only re-run TTS when audio not cached.
        if self._tm is not None:
            tm_hit = self._tm.lookup(vi_text_acc, target)
            if tm_hit is not None:
                cached_audio = tm_hit.get("audio")
                if cached_audio is not None:
                    # Full hit: reuse target text + audio.
                    yield {
                        "stage": "tm_full_hit",
                        "vi_text": vi_text_acc,
                        "tgt_text": tm_hit["tgt_text"],
                        "tgt_audio": cached_audio[1],
                        "sr": cached_audio[0],
                        "latency": {"cache_ms": round(cache_ms, 2),
                                    "asr_ms": round(asr_ms, 1),
                                    "mt_ms": 0, "tts_ms": 0,
                                    "tm_cosine": round(tm_hit["cosine"], 4),
                                    "total_ms": round((cache_ms + asr_ms), 1)},
                        "cache": {"hit": False, "cosine": None, "tm_hit": True,
                                  "tm_cosine": round(tm_hit["cosine"], 4),
                                  "stats": self._utt_cache.summary() if self._utt_cache else None,
                                  "tm_stats": self._tm.summary()},
                    }
                    return
                # Partial hit: reuse target text, re-run TTS only.
                yield {
                    "stage": "mt_partial",
                    "vi_text": vi_text_acc,
                    "tgt_text": tm_hit["tgt_text"],
                    "tgt_audio": None,
                    "sr": 16000,
                    "latency": {"cache_ms": round(cache_ms, 2),
                                "asr_ms": round(asr_ms, 1),
                                "mt_ms": 0, "tts_ms": 0,
                                "tm_cosine": round(tm_hit["cosine"], 4),
                                "total_ms": round((cache_ms + asr_ms), 1)},
                    "cache": {"hit": False, "cosine": None, "tm_hit": True,
                              "tm_cosine": round(tm_hit["cosine"], 4),
                              "stats": self._utt_cache.summary() if self._utt_cache else None,
                              "tm_stats": self._tm.summary()},
                }
                # Synthesize cached translation, then emit done.
                t_tts_start = _t.perf_counter()
                tgt_audio, out_sr = self.synthesize(tm_hit["tgt_text"], target)
                tts_ms = (_t.perf_counter() - t_tts_start) * 1000
                total_ms = (_t.perf_counter() - t0) * 1000
                # Backfill audio into TM for future full hits.
                self._tm.put(vi_text_acc, tm_hit["tgt_text"], target, audio=(out_sr, tgt_audio))
                if self._utt_cache is not None and fp is not None:
                    self._utt_cache.put(fp, target, {
                        "vi": vi_text_acc, "tgt": tm_hit["tgt_text"], "audio": (out_sr, tgt_audio),
                    })
                yield {
                    "stage": "done",
                    "vi_text": vi_text_acc,
                    "tgt_text": tm_hit["tgt_text"],
                    "tgt_audio": tgt_audio,
                    "sr": out_sr,
                    "latency": {"cache_ms": round(cache_ms, 2),
                                "asr_ms": round(asr_ms, 1),
                                "mt_ms": 0, "tts_ms": round(tts_ms, 1),
                                "tm_cosine": round(tm_hit["cosine"], 4),
                                "total_ms": round(total_ms, 1)},
                    "cache": {"hit": False, "cosine": None, "tm_hit": True,
                              "tm_cosine": round(tm_hit["cosine"], 4),
                              "stats": self._utt_cache.summary() if self._utt_cache else None,
                              "tm_stats": self._tm.summary()},
                }
                return

        # ----- MT sentence-by-sentence streaming -----
        self._ensure_mt()
        sentences = _split_sentences(vi_text_acc)
        tgt_acc = ""
        t_mt_start = _t.perf_counter()
        for sent in sentences:
            if not sent.strip():
                continue
            tr = self.translate(sent, target, source="vi")
            tgt_acc = (tgt_acc + " " + tr).strip() if tgt_acc else tr
            mt_ms_so_far = (_t.perf_counter() - t_mt_start) * 1000
            yield {
                "stage": "mt_partial",
                "vi_text": vi_text_acc,
                "tgt_text": tgt_acc,
                "tgt_audio": None,
                "sr": 16000,
                "latency": {"cache_ms": round(cache_ms, 2),
                            "asr_ms": round(asr_ms, 1),
                            "mt_ms": round(mt_ms_so_far, 1),
                            "tts_ms": 0,
                            "total_ms": round((cache_ms + asr_ms + mt_ms_so_far), 1)},
                "cache": {"hit": False, "cosine": None,
                          "stats": self._utt_cache.summary() if self._utt_cache else None},
            }
        mt_ms = (_t.perf_counter() - t_mt_start) * 1000

        # ----- TTS on the full target text (could be streamed too but VITS doesn't expose chunk API cheaply) -----
        t_tts_start = _t.perf_counter()
        tgt_audio, out_sr = self.synthesize(tgt_acc, target)
        tts_ms = (_t.perf_counter() - t_tts_start) * 1000
        total_ms = (_t.perf_counter() - t0) * 1000

        # Populate both caches for future hits
        if self._utt_cache is not None and fp is not None:
            self._utt_cache.put(fp, target, {
                "vi": vi_text_acc, "tgt": tgt_acc, "audio": (out_sr, tgt_audio),
            })
        if self._tm is not None:
            self._tm.put(vi_text_acc, tgt_acc, target, audio=(out_sr, tgt_audio))

        yield {
            "stage": "done",
            "vi_text": vi_text_acc,
            "tgt_text": tgt_acc,
            "tgt_audio": tgt_audio,
            "sr": out_sr,
            "latency": {"cache_ms": round(cache_ms, 2),
                        "asr_ms": round(asr_ms, 1),
                        "mt_ms": round(mt_ms, 1),
                        "tts_ms": round(tts_ms, 1),
                        "total_ms": round(total_ms, 1)},
            "cache": {"hit": False, "cosine": None, "tm_hit": False,
                      "stats": self._utt_cache.summary() if self._utt_cache else None,
                      "tm_stats": self._tm.summary() if self._tm else None},
        }

    # ---------------- End-to-end (non-streaming) ----------------
    def run(self, audio: np.ndarray, sr: int, target: str) -> dict:
        # Cache check FIRST (before any model work)
        cache_hit = False
        cache_cos = None
        cache_ms = 0.0
        if self._utt_cache is not None:
            tc0 = time.perf_counter()
            fp = self._utt_cache.fingerprint(audio, sr)
            entry = self._utt_cache.lookup(fp, target)
            cache_ms = (time.perf_counter() - tc0) * 1000
            if entry is not None:
                cache_hit = True
                cache_cos = entry["cosine"]
                res = entry["result"]
                return {
                    "vi_text":   res["vi"],
                    "tgt_text":  res["tgt"],
                    "tgt_audio": res["audio"][1],
                    "sr":        res["audio"][0],
                    "latency":   {"asr_ms": 0, "mt_ms": 0, "tts_ms": 0,
                                  "cache_ms": round(cache_ms, 2),
                                  "total_ms": round(cache_ms, 2)},
                    "cache":     {"hit": True, "cosine": round(cache_cos, 4),
                                  "stats": self._utt_cache.summary()},
                }

        # Cache miss: run full cascade
        self._bg_preload_tts(target)
        t0 = time.perf_counter()
        vi = self.transcribe(audio, sr)
        t1 = time.perf_counter()
        tgt = self.translate(vi, target, source="vi")
        t2 = time.perf_counter()
        tgt_audio, out_sr = self.synthesize(tgt, target)
        t3 = time.perf_counter()

        # Populate cache for future hits
        if self._utt_cache is not None:
            self._utt_cache.put(fp, target, {
                "vi": vi, "tgt": tgt, "audio": (out_sr, tgt_audio),
            })

        return {
            "vi_text":   vi,
            "tgt_text":  tgt,
            "tgt_audio": tgt_audio,
            "sr":        out_sr,
            "latency":   {"asr_ms":   (t1 - t0) * 1000,
                          "mt_ms":    (t2 - t1) * 1000,
                          "tts_ms":   (t3 - t2) * 1000,
                          "cache_ms": round(cache_ms, 2),
                          "total_ms": (t3 - t0) * 1000 + cache_ms},
            "cache":     {"hit": False, "cosine": None,
                          "stats": self._utt_cache.summary() if self._utt_cache else None},
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio",  required=True)
    p.add_argument("--target", choices=["en", "zh", "ko", "vi"], default="en")
    p.add_argument("--out",    default="prototype/out_fast.wav")
    args = p.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"[input] {args.audio}  duration={duration:.2f}s  sr={sr}")

    pipe = FastCascadeS2ST()
    res = pipe.run(audio, sr, args.target)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), res["tgt_audio"], res["sr"])
    out_dur = len(res["tgt_audio"]) / max(res["sr"], 1)

    print()
    print(f"[asr]   vi : {res['vi_text']}")
    print(f"[mt]    {args.target:>2} : {res['tgt_text']}")
    print(f"[tts]   wrote {out}  ({out_dur:.2f}s @ {res['sr']} Hz)")
    print()
    lat = res["latency"]
    print(f"[latency] asr={lat['asr_ms']:>7.0f} ms   mt={lat['mt_ms']:>7.0f} ms   "
          f"tts={lat['tts_ms']:>7.0f} ms   total={lat['total_ms']:>7.0f} ms")
    rtf = lat["total_ms"] / 1000 / duration
    print(f"[rtf]   real-time factor = {rtf:.2f}x  (>1 means slower than real-time on this device)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
