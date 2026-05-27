"""
Tone-conditioned ASR decoder early-exit — prototype + honest evaluation.

Motivation (paper §3.9 candidate):
  Whisper's autoregressive decoder runs to `max_new_tokens` (default 256) or EOS.
  For short Vi industrial phrases (3-10 syllables) the lexical content is decided
  within the first ~20 tokens, but EOS probability builds slowly so the decoder
  keeps emitting timestamp/padding tokens. We want to *force* EOS once the
  emitted prefix is high-confidence and stable.

Mechanism (proposed):
  Per step, monitor token-distribution entropy.
  If entropy < entropy_thr for K consecutive steps AND mean logprob > logprob_thr,
  force EOS.

Reality with faster-whisper (CTranslate2):
  We cannot easily hook into per-step logits without forking the C++ decoder.
  Two viable evaluation surfaces remain:

    (A) POST-HOC: run normal transcribe(word_timestamps=True), use per-word
        `.probability` as an entropy proxy (entropy_proxy_word = -log p_word).
        Then simulate: smallest K-word prefix whose normalised text already
        equals the final transcript → that's how many words the decoder
        *needed*. Savings = (full_tokens - tokens_in_K_prefix) / full_tokens.

    (B) GROUND-TRUTH LATENCY KNOB: faster-whisper's `transcribe(max_new_tokens=...)`
        is the *real* early-exit. Sweep max_new_tokens ∈ {16, 32, 64, 128, 256}
        and measure (latency, WER) per cap. Find the smallest cap that does
        not degrade WER. This is what a deployable implementation would do
        via dynamic cap selection.

We report (A) for the "how much could we save" estimate and (B) for the
"what does this actually buy us in seconds" ground-truth latency curve.

Honest caveat: (A) over-estimates savings because in real autoregressive
decoding the model doesn't know it's about to emit redundant timestamp
tokens. (B) is the ceiling of what a static cap can deliver — a dynamic
entropy-triggered early-exit would sit between (A) and (B).

Usage:
  python prototype/tone_early_exit.py --n 10
  python prototype/tone_early_exit.py --n 20 --caps 16 32 64 128 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                             # noqa: E402
import soundfile as sf                                          # noqa: E402


MODEL_DIR = PROJECT / "data" / "raw" / "models_ct2" / "phowhisper-small-int8"
VIVOS_TEST = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"
RESULTS = PROJECT / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)


# --------------------------- text utils ---------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _norm(text: str) -> str:
    """Lowercase, strip diacritic-preserving punctuation, collapse whitespace."""
    if not text: return ""
    t = text.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _norm_tokens(text: str) -> list[str]:
    return _norm(text).split()


def _wer(ref: str, hyp: str) -> float:
    """Standard word-error-rate via Levenshtein on whitespace tokens."""
    r = _norm_tokens(ref)
    h = _norm_tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if r[i-1] == h[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


# --------------------------- data ---------------------------

def _load_prompts() -> dict[str, str]:
    out: dict[str, str] = {}
    pf = VIVOS_TEST / "prompts.txt"
    if not pf.exists():
        return out
    with open(pf, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0].strip()] = parts[1].strip()
    return out


def _gather_vivos_test(n: int) -> list[Path]:
    waves = VIVOS_TEST / "waves"
    if not waves.exists():
        return []
    out: list[Path] = []
    speakers = sorted([d for d in waves.iterdir() if d.is_dir()])
    # Round-robin across speakers for variety.
    per_spk = max(1, n // max(1, len(speakers)))
    for spk in speakers:
        for w in sorted(spk.glob("*.wav"))[:per_spk]:
            out.append(w)
            if len(out) >= n: return out
    return out


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


# --------------------------- ASR wrappers ---------------------------

def _build_model():
    from faster_whisper import WhisperModel
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"PhoWhisper-small INT8 not found at {MODEL_DIR}")
    print(f"[load] {MODEL_DIR}  compute=int8")
    return WhisperModel(
        str(MODEL_DIR), device="cpu", compute_type="int8",
        cpu_threads=0, num_workers=1,
    )


def _transcribe_full(model, audio: np.ndarray, max_new_tokens: int | None):
    """Single transcribe call. Returns (segments_list, info, elapsed_s).
    word_timestamps=True forces faster-whisper to emit per-word probabilities."""
    t0 = time.perf_counter()
    seg_gen, info = model.transcribe(
        audio,
        language="vi",
        beam_size=1,
        vad_filter=False,                    # we want bare decode timing
        condition_on_previous_text=False,
        without_timestamps=False,            # need timestamps so .words populates
        word_timestamps=True,
        max_new_tokens=max_new_tokens,
    )
    segments = list(seg_gen)
    elapsed = time.perf_counter() - t0
    return segments, info, elapsed


def _full_text(segments) -> str:
    return " ".join(s.text for s in segments).strip()


def _flatten_words(segments) -> list[tuple[str, float]]:
    out = []
    for s in segments:
        if not s.words: continue
        for w in s.words:
            out.append((w.word, float(w.probability)))
    return out


def _count_tokens(segments) -> int:
    return sum(len(s.tokens) for s in segments)


# --------------------------- (A) post-hoc simulation ---------------------------

def posthoc_equivalent_K(words: list[tuple[str, float]], final_text: str) -> int:
    """Smallest K such that normalised text of first K words == normalised final_text.
    If never matches, return len(words)."""
    target = _norm(final_text)
    if not target or not words:
        return len(words)
    target_toks = target.split()
    cum = ""
    for K in range(1, len(words) + 1):
        cum_txt = " ".join(w for w, _ in words[:K])
        cum_norm = _norm(cum_txt).split()
        if cum_norm == target_toks:
            return K
    return len(words)


def entropy_proxy_at(words: list[tuple[str, float]], K: int) -> dict:
    """Stats over first K word-probabilities (entropy proxy = -log p)."""
    if K <= 0 or not words:
        return {"K": 0, "mean_logprob": None, "mean_entropy_proxy": None,
                "min_prob": None}
    probs = [p for _, p in words[:K]]
    eps = 1e-12
    logps = [math.log(max(p, eps)) for p in probs]
    return {
        "K": K,
        "mean_logprob": sum(logps) / len(logps),
        "mean_entropy_proxy": -sum(logps) / len(logps),
        "min_prob": min(probs),
    }


# --------------------------- main eval ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--caps", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    ap.add_argument("--out", default=str(RESULTS / "tone_early_exit_eval.json"))
    args = ap.parse_args()

    prompts = _load_prompts()
    utts = _gather_vivos_test(args.n)
    if not utts:
        print(f"[err] no VIVOS test wavs under {VIVOS_TEST}")
        return 1
    print(f"[data] {len(utts)} VIVOS test utterances")
    print(f"[data] {len(prompts)} prompts loaded")

    model = _build_model()

    per_utt: list[dict] = []

    # We use the unrestricted cap=256 run as the "reference" full decode.
    # Then we re-run the same audio at each smaller cap to measure latency+WER.

    for i, wav in enumerate(utts):
        utt_id = wav.stem
        ref = prompts.get(utt_id, "")
        audio, sr = _load_audio(wav)
        if sr != 16000:
            tlen = int(round(len(audio) * 16000 / sr))
            audio = np.interp(np.linspace(0, len(audio), tlen, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)

        # Reference run (cap=256, no early exit).
        segs_ref, _info, elapsed_ref = _transcribe_full(model, audio, max_new_tokens=256)
        text_ref = _full_text(segs_ref)
        n_tokens_ref = _count_tokens(segs_ref)
        words = _flatten_words(segs_ref)

        # (A) post-hoc equivalent K
        K = posthoc_equivalent_K(words, text_ref)
        ep_at_K = entropy_proxy_at(words, K)
        savings_words = (len(words) - K) / max(1, len(words))
        # tokens_in_first_K_words is unknown without per-token alignment; approximate
        # via proportional share of decoded tokens.
        tokens_K_approx = int(round(n_tokens_ref * K / max(1, len(words))))
        savings_tokens_approx = (n_tokens_ref - tokens_K_approx) / max(1, n_tokens_ref)

        # WER of reference vs ground truth (sanity)
        wer_ref = _wer(ref, text_ref) if ref else None

        # (B) cap sweep
        cap_results = []
        for cap in args.caps:
            if cap >= 256:
                # reuse reference run to avoid duplicate work
                cap_results.append({
                    "cap": cap,
                    "elapsed_s": elapsed_ref,
                    "n_tokens_decoded": n_tokens_ref,
                    "text": text_ref,
                    "wer_vs_ref": 0.0,
                    "wer_vs_gt": wer_ref,
                })
                continue
            segs_c, _ic, elapsed_c = _transcribe_full(model, audio, max_new_tokens=cap)
            text_c = _full_text(segs_c)
            n_tok_c = _count_tokens(segs_c)
            cap_results.append({
                "cap": cap,
                "elapsed_s": elapsed_c,
                "n_tokens_decoded": n_tok_c,
                "text": text_c,
                "wer_vs_ref": _wer(text_ref, text_c),
                "wer_vs_gt": _wer(ref, text_c) if ref else None,
            })

        entry = {
            "utt_id": utt_id,
            "ref_text_gt": ref,
            "ref_text_hyp": text_ref,
            "wer_ref_vs_gt": wer_ref,
            "audio_s": len(audio) / 16000.0,
            "full_decode": {
                "elapsed_s": elapsed_ref,
                "n_tokens_decoded": n_tokens_ref,
                "n_words": len(words),
            },
            "posthoc_simulation_A": {
                "equivalent_K_words": K,
                "tokens_in_first_K_words_approx": tokens_K_approx,
                "savings_pct_words": 100.0 * savings_words,
                "savings_pct_tokens_approx": 100.0 * savings_tokens_approx,
                "mean_logprob_first_K": ep_at_K["mean_logprob"],
                "mean_entropy_proxy_first_K": ep_at_K["mean_entropy_proxy"],
                "min_word_prob_first_K": ep_at_K["min_prob"],
            },
            "cap_sweep_B": cap_results,
        }
        per_utt.append(entry)

        print(f"  [{i+1:2d}/{len(utts)}] {utt_id}  "
              f"audio={entry['audio_s']:.2f}s  "
              f"toks={n_tokens_ref:3d}  words={len(words):2d}  K={K:2d}  "
              f"save~{100*savings_tokens_approx:5.1f}%  "
              f"ref={elapsed_ref:.2f}s  WER_gt={wer_ref}")

    # ---------- aggregate ----------
    def _agg(values: list[float]) -> dict:
        clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not clean:
            return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
        clean_sorted = sorted(clean)
        return {
            "mean":   statistics.mean(clean),
            "median": statistics.median(clean),
            "p95":    clean_sorted[min(len(clean_sorted)-1, int(round(0.95 * (len(clean_sorted)-1))))],
            "min":    clean_sorted[0],
            "max":    clean_sorted[-1],
        }

    full_tokens   = [u["full_decode"]["n_tokens_decoded"] for u in per_utt]
    equiv_K       = [u["posthoc_simulation_A"]["equivalent_K_words"] for u in per_utt]
    save_tokens   = [u["posthoc_simulation_A"]["savings_pct_tokens_approx"] for u in per_utt]
    save_words    = [u["posthoc_simulation_A"]["savings_pct_words"] for u in per_utt]
    full_latency  = [u["full_decode"]["elapsed_s"] for u in per_utt]
    wer_gt        = [u["wer_ref_vs_gt"] for u in per_utt if u["wer_ref_vs_gt"] is not None]

    cap_agg = {}
    for cap in args.caps:
        lats = []
        wers_ref = []
        wers_gt  = []
        toks     = []
        for u in per_utt:
            for cr in u["cap_sweep_B"]:
                if cr["cap"] == cap:
                    lats.append(cr["elapsed_s"])
                    wers_ref.append(cr["wer_vs_ref"])
                    if cr["wer_vs_gt"] is not None:
                        wers_gt.append(cr["wer_vs_gt"])
                    toks.append(cr["n_tokens_decoded"])
        # latency savings vs cap=256 mean
        ref_mean_lat = statistics.mean([u["full_decode"]["elapsed_s"] for u in per_utt])
        cap_agg[str(cap)] = {
            "latency_s": _agg(lats),
            "latency_savings_pct_vs_256_mean": 100.0 * (ref_mean_lat - statistics.mean(lats)) / ref_mean_lat if lats else None,
            "wer_vs_ref":  _agg(wers_ref),
            "wer_vs_gt":   _agg(wers_gt),
            "n_tokens_decoded": _agg(toks),
        }

    aggregate = {
        "n_utts": len(per_utt),
        "full_decode_tokens":        _agg(full_tokens),
        "full_decode_latency_s":     _agg(full_latency),
        "posthoc_equivalent_K":      _agg(equiv_K),
        "posthoc_savings_pct_tokens_approx": _agg(save_tokens),
        "posthoc_savings_pct_words":         _agg(save_words),
        "wer_full_vs_gt":            _agg(wer_gt),
        "cap_sweep_B": cap_agg,
    }

    out_obj = {
        "config": {
            "model": str(MODEL_DIR),
            "n_utts": len(per_utt),
            "caps": args.caps,
            "note": "Simulation A is a post-hoc upper bound on tokens we could "
                    "skip if a perfect oracle told us when the model is done. "
                    "Simulation B (cap sweep) is the deployable ground truth.",
        },
        "per_utt": per_utt,
        "aggregate": aggregate,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"\n[save] {args.out}")

    # ---------- pretty print ----------
    print("\n" + "=" * 70)
    print("AGGREGATE — tone_early_exit_eval")
    print("=" * 70)
    print(f"  n_utts                        = {aggregate['n_utts']}")
    ft = aggregate["full_decode_tokens"]; fl = aggregate["full_decode_latency_s"]
    print(f"  full decode tokens            : median={ft['median']:.0f}  p95={ft['p95']:.0f}  max={ft['max']:.0f}")
    print(f"  full decode latency (s)       : median={fl['median']:.3f}  p95={fl['p95']:.3f}")
    print(f"  WER full vs ground truth      : mean={aggregate['wer_full_vs_gt']['mean']}")
    print()
    print("  (A) POST-HOC SIMULATION — upper bound on early-exit savings")
    eK = aggregate["posthoc_equivalent_K"]; sT = aggregate["posthoc_savings_pct_tokens_approx"]
    print(f"    equivalent_K words          : median={eK['median']:.1f}  p95={eK['p95']:.1f}")
    print(f"    token savings approx (%)    : median={sT['median']:.1f}  p95={sT['p95']:.1f}  mean={sT['mean']:.1f}")
    print()
    print("  (B) MAX_NEW_TOKENS CAP SWEEP — deployable ground truth")
    print(f"    {'cap':>5} {'lat_med':>9} {'lat_p95':>9} {'sav_%':>7} {'WER_ref':>9} {'WER_gt':>9}")
    for cap in args.caps:
        c = cap_agg[str(cap)]
        lat_med = c["latency_s"]["median"]; lat_p95 = c["latency_s"]["p95"]
        sav = c["latency_savings_pct_vs_256_mean"]
        wer_r = c["wer_vs_ref"]["mean"]; wer_g = c["wer_vs_gt"]["mean"]
        print(f"    {cap:>5} {lat_med:>9.3f} {lat_p95:>9.3f} "
              f"{sav if sav is not None else 0.0:>7.1f} "
              f"{wer_r if wer_r is not None else 0.0:>9.3f} "
              f"{wer_g if wer_g is not None else 0.0:>9.3f}")

    print()
    print("Caveats:")
    print("  - (A) over-estimates: an oracle early-exit isn't achievable; faster-whisper")
    print("    has no per-step entropy hook exposed at the Python layer.")
    print("  - (B) is what we can ship today via dynamic max_new_tokens selection.")
    print("  - PhoWhisper-small often emits multi-token timestamp+text frames per word,")
    print("    so 'tokens per word' is an approximation (we report words for safety).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
