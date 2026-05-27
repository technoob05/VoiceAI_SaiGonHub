"""
exp00 — Tone-Error-Rate (TER) metric for Vietnamese ASR
Spine: tone-aware mixed-precision quantization for Vietnamese on-device S2ST.
Goal of this script:
  (1) Implement TER: syllable-aligned tonal-mismatch rate, isolated from base-segment errors.
  (2) Sanity-check on VIVOS test split with PhoWhisper-small FP32 (single Kaggle T4/P100 run).
  (3) Emit Results/exp00_ter_metric.json so every later quant experiment shares this metric.

Self-contained. No imports from a shared lib. Copy-paste to Kaggle.

Kaggle setup:
  - Add Data → attach `voiceai-tier1-vi` (VIVOS).
  - Add Data → attach the PhoWhisper-small model snapshot (or set Internet ON to fetch from HF).
  - Update DATA_ROOT and MODEL_ID below if paths differ.

Output:
  Results/exp00_ter_metric.json   ← config + metrics + per-tone confusion matrix.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable
import json
import os
import re
import sys
import time
import unicodedata

# Ensure stdout/stderr can print Vietnamese on Windows (cp1252 default).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try: _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

# ============================================================ Config

@dataclass
class Config:
    EXP_ID:      str = "exp00"
    MODEL_ID:    str = "vinai/PhoWhisper-small"
    DATASET_ID:  str = "AILAB-VNUHCM/vivos"
    SPLIT:       str = "test"
    MAX_SAMPLES: int = 500                  # set None for full split; override via VOICEAI_MAX_SAMPLES env
    BATCH_SIZE:  int = 8
    NUM_WORKERS: int = 2
    SEED:        int = 42
    LANGUAGE:    str = "vi"                  # Whisper decoder language token
    OUTPUT_DIR:  str = "Results"
    DEVICE:      str = "auto"                # "auto" | "cuda" | "cpu" | "mps"
    DTYPE:       str = "auto"                # "auto" picks float32 on CPU, float16 on CUDA
    # Local dev (Windows): data/processed/vivos  -> contains vivos/test/{waves,prompts.txt}
    # Kaggle:               /kaggle/input/voiceai-tier1-vi/vivos/test/...
    DATA_ROOT:   str = "data/processed/vivos"

cfg = Config()
if os.environ.get("VOICEAI_MAX_SAMPLES"):
    cfg.MAX_SAMPLES = int(os.environ["VOICEAI_MAX_SAMPLES"])


# ============================================================ Vietnamese tone extraction
# 6 tones. Encoded via Unicode combining marks on the syllable nucleus.
#   ngang  (level, no mark)            tone_id = 0
#   huyền  (falling, grave)            tone_id = 1   U+0300
#   sắc    (rising, acute)             tone_id = 2   U+0301
#   hỏi    (dipping, hook above)       tone_id = 3   U+0309
#   ngã    (creaky rising, tilde)      tone_id = 4   U+0303
#   nặng   (low rising, dot below)     tone_id = 5   U+0323

_TONE_MARKS: dict[str, int] = {
    "̀": 1, "́": 2, "̉": 3, "̃": 4, "̣": 5,
}
TONE_NAMES = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]


def _strip_diacritics(s: str) -> str:
    """Return base form of `s` with all combining marks removed (also lower-cases)."""
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def syllable_tone(syllable: str) -> int:
    """Extract tone id 0..5 from a Vietnamese syllable.

    Vietnamese marks tone with at most one combining diacritic on the nucleus.
    If multiple tone marks appear (unusual / typo), the FIRST one wins for determinism.
    Non-Vietnamese / digit tokens return 0 (treated as 'no tone' — they will be skipped
    from TER aggregation downstream).
    """
    nfd = unicodedata.normalize("NFD", syllable)
    for ch in nfd:
        if ch in _TONE_MARKS:
            return _TONE_MARKS[ch]
    return 0


_PUNCT_RE = re.compile(r"[^\w\sÀ-ỹ]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")

def normalize_vi(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, return Vietnamese syllable tokens.

    Vietnamese orthography puts a space between every syllable. So tokens here ARE syllables.
    """
    if text is None:
        return []
    t = _PUNCT_RE.sub(" ", text.lower())
    t = _WS_RE.sub(" ", t).strip()
    return [s for s in t.split(" ") if s]


# ============================================================ Alignment + WER/TER

def _levenshtein_ops(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """Return list of edit ops as (op, ref_idx, hyp_idx). op ∈ {'=', 'S', 'I', 'D'}."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # match/sub
            )
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("=", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("S", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", i - 1, -1));     i -= 1
        else:
            ops.append(("I", -1, j - 1));     j -= 1
    ops.reverse()
    return ops


@dataclass
class ScoreState:
    ref_syllables: int = 0
    hyp_syllables: int = 0
    subs: int = 0
    ins:  int = 0
    dels: int = 0
    base_matches: int = 0
    tone_only_errors: int = 0
    base_errors: int = 0
    confusion: dict = field(default_factory=dict)  # f"{ref_tone}->{hyp_tone}" → count

    def add(self, ref: list[str], hyp: list[str]) -> None:
        self.ref_syllables += len(ref)
        self.hyp_syllables += len(hyp)
        for op, ri, hi in _levenshtein_ops(ref, hyp):
            if op == "=":
                self.base_matches += 1
            elif op == "S":
                self.subs += 1
                r = ref[ri]; h = hyp[hi]
                if _strip_diacritics(r) == _strip_diacritics(h):
                    self.tone_only_errors += 1
                    rt = syllable_tone(r); ht = syllable_tone(h)
                    key = f"{TONE_NAMES[rt]}->{TONE_NAMES[ht]}"
                    self.confusion[key] = self.confusion.get(key, 0) + 1
                else:
                    self.base_errors += 1
            elif op == "I":
                self.ins += 1; self.base_errors += 1
            elif op == "D":
                self.dels += 1; self.base_errors += 1

    def summary(self) -> dict:
        n = max(self.ref_syllables, 1)
        return {
            "ref_syllables":     self.ref_syllables,
            "hyp_syllables":     self.hyp_syllables,
            "subs":              self.subs,
            "ins":               self.ins,
            "dels":              self.dels,
            "base_matches":      self.base_matches,
            "tone_only_errors":  self.tone_only_errors,
            "base_errors":       self.base_errors,
            # Vietnamese is syllable-tokenized, so "WER" here = syllable error rate (SER).
            "wer":               (self.subs + self.ins + self.dels) / n,
            # TER: syllables where the base segment is correct but the tone is wrong,
            #      normalized by reference syllable count. Isolates the tonal information loss.
            "ter":               self.tone_only_errors / n,
            # TER conditional on base-correct: how often, when we DO get the segment right,
            #      do we still get the tone wrong? More sensitive to quantization-induced tone collapse.
            "ter_given_base_ok": self.tone_only_errors / max(self.base_matches + self.tone_only_errors, 1),
            "confusion":         dict(sorted(self.confusion.items(), key=lambda kv: -kv[1])),
        }


# ============================================================ Self-test (always run)
# Synthetic gold-vs-pred pairs to verify the metric reacts as expected before any ASR is touched.

def _self_test() -> None:
    pairs = [
        ("xin chào",        "xin chao",        "tone-only error, 1 syllable affected"),
        ("xin chào các bạn", "xin chào các bạn", "perfect"),
        ("xin chào các bạn", "xin chao cac ban", "all tones stripped"),
        ("anh đã ăn cơm",   "anh đa ăn cơm",   "1 tone error"),
        ("anh đã ăn cơm",   "anh đã ăn cam",   "base error (cơm→cam, not tone)"),
    ]
    print("[self-test] sanity-checking TER on synthetic pairs:")
    for ref, hyp, note in pairs:
        st = ScoreState()
        st.add(normalize_vi(ref), normalize_vi(hyp))
        s = st.summary()
        print(f"  ref={ref!r} hyp={hyp!r}  wer={s['wer']:.2f}  ter={s['ter']:.2f}  base_err={s['base_errors']}  tone_only={s['tone_only_errors']}  [{note}]")
    print()


# ============================================================ Main eval loop

def _load_vivos_local(split: str, data_root: str) -> Iterable[dict]:
    """Walk an extracted VIVOS tree. Expected layout:
        <data_root>/vivos/<split>/prompts.txt          (lines: "<utt_id> <UPPERCASE_VIETNAMESE>")
        <data_root>/vivos/<split>/waves/<spk>/<utt_id>.wav  (16 kHz mono)
    Also tolerates the case where <data_root> already points at the inner `vivos/` dir.
    """
    import soundfile as sf
    base = Path(data_root)
    candidates = [base / "vivos" / split, base / split, base / "vivos" / "vivos" / split]
    cand = next((c for c in candidates if (c / "prompts.txt").exists()), None)
    if cand is None:
        raise FileNotFoundError(f"VIVOS prompts.txt not found under any of: {candidates}")

    prompts = {}
    for line in (cand / "prompts.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        prompts[utt_id.strip()] = text.strip()
    waves_root = cand / "waves"
    print(f"[data] VIVOS local: {cand}  utts={len(prompts)}  speakers={sum(1 for _ in waves_root.iterdir())}")

    for utt_id, text in prompts.items():
        spk = utt_id.split("_")[0]
        wav = waves_root / spk / f"{utt_id}.wav"
        if not wav.exists():
            continue
        audio, sr = sf.read(str(wav), dtype="float32")
        yield {"audio": audio, "sr": sr, "text": text}


def _load_fleurs_parquet(lang_config: str, split: str, data_root: str) -> Iterable[dict]:
    """Walk a FLEURS parquet snapshot. Expected layout:
        <data_root>/parquet-data/<lang_config>/<split>-00000-of-00001.parquet
    """
    import io
    import soundfile as sf
    import pyarrow.parquet as pq
    base = Path(data_root)
    candidates = [base / "parquet-data" / lang_config, base / lang_config]
    cand = next((c for c in candidates if c.exists()), None)
    if cand is None:
        raise FileNotFoundError(f"FLEURS {lang_config}/{split} not found under: {candidates}")
    files = sorted(cand.glob(f"{split}-*-of-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards for {lang_config}/{split} under {cand}")
    for f in files:
        table = pq.read_table(f)
        rows = table.to_pylist()
        print(f"[data] FLEURS {lang_config}/{split} from {f.name}, n={len(rows)}")
        for row in rows:
            audio_bytes = row.get("audio", {}).get("bytes") if isinstance(row.get("audio"), dict) else None
            if audio_bytes is None:
                continue
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            yield {"audio": audio, "sr": sr,
                   "text": row.get("transcription") or row.get("raw_transcription") or ""}


def _load_dataset_iter(dsid: str, split: str, data_root: str) -> Iterable[dict]:
    """Dispatch to a local-walker based on the dataset id, since `datasets.load_dataset`
    no longer supports the legacy loading scripts these datasets ship with.
    """
    if "vivos" in dsid.lower():
        yield from _load_vivos_local(split, data_root)
    elif "fleurs" in dsid.lower():
        # dsid like "fleurs:vi_vn" or DATASET_ID + LANG_CONFIG separately. Default to vi_vn.
        lang = "vi_vn"
        if ":" in dsid:
            _, lang = dsid.split(":", 1)
        yield from _load_fleurs_parquet(lang, split, data_root)
    else:
        # Fallback: try modern parquet load_dataset (only works for already-converted datasets).
        from datasets import load_dataset, Audio
        ds = load_dataset(dsid, split=split).cast_column("audio", Audio(sampling_rate=16000))
        for row in ds:
            yield {"audio": row["audio"]["array"], "sr": row["audio"]["sampling_rate"],
                   "text": row.get("sentence") or row.get("text") or row.get("transcription") or ""}


def _run_model_eval() -> dict:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    if cfg.DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg.DEVICE if (cfg.DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    if cfg.DTYPE == "auto":
        dtype_name = "float16" if device == "cuda" else "float32"
    else:
        dtype_name = cfg.DTYPE
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    print(f"[model] loading {cfg.MODEL_ID} on {device} ({dtype_name})")
    proc  = WhisperProcessor.from_pretrained(cfg.MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.MODEL_ID, torch_dtype=dtype).to(device).eval()
    model.generation_config.language = cfg.LANGUAGE
    model.generation_config.task     = "transcribe"

    state = ScoreState()
    n_seen = 0
    latencies_ms: list[float] = []

    for row in _load_dataset_iter(cfg.DATASET_ID, cfg.SPLIT, cfg.DATA_ROOT):
        if cfg.MAX_SAMPLES and n_seen >= cfg.MAX_SAMPLES:
            break
        inputs = proc(row["audio"], sampling_rate=row["sr"], return_tensors="pt")
        inputs = {k: v.to(device, dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        t0 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=256)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
        state.add(normalize_vi(row["text"]), normalize_vi(hyp))
        n_seen += 1
        if n_seen % 50 == 0:
            print(f"  [eval] {n_seen} done, running ter={state.summary()['ter']:.4f}")

    out = state.summary()
    if latencies_ms:
        latencies_ms.sort()
        out["latency_ms_p50"] = latencies_ms[len(latencies_ms) // 2]
        out["latency_ms_p95"] = latencies_ms[int(len(latencies_ms) * 0.95)]
        out["latency_ms_p99"] = latencies_ms[int(len(latencies_ms) * 0.99)]
    out["n_test_utterances"] = n_seen

    # Approx model size in MB (parameter bytes only).
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    out["model_size_mb"] = n_bytes / (1024 * 1024)
    return out


def main() -> int:
    _self_test()

    metrics: dict
    try:
        metrics = _run_model_eval()
    except Exception as e:
        print(f"[warn] model eval skipped: {type(e).__name__}: {e}")
        print("[warn] writing self-test-only result so the metric implementation is still verifiable")
        st = ScoreState()
        for ref, hyp in [("xin chào", "xin chao"), ("anh đã ăn cơm", "anh đa ăn cơm")]:
            st.add(normalize_vi(ref), normalize_vi(hyp))
        metrics = st.summary()
        metrics["n_test_utterances"] = 0
        metrics["model_size_mb"] = None
        metrics["mode"] = "self-test-only"

    result = {
        "config": asdict(cfg),
        "metrics": metrics,
        "wall_clock_seconds": round(time.time() - START, 1),
        "host": os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or "local",
    }
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.OUTPUT_DIR) / f"{cfg.EXP_ID}_ter_metric.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


START = time.time()
if __name__ == "__main__":
    sys.exit(main())
