"""
Side-by-side benchmark: UtteranceCache (Python loop) vs VectorizedUtteranceCache (single matmul).

Mirrors cache_scalability_bench.py methodology:
  - Fill cache to N entries from VIVOS train wavs (with jitter beyond pool size).
  - 200 lookup queries from VIVOS test wavs, 5 warmup queries discarded.
  - Time the comparison step alone (fingerprint pre-computed outside the timer).
  - Verify exact equivalence: for the same query, both caches must agree on
    the chosen entry id (when both hit) and cosine within 1e-5.

Output: Results/vectorized_cache_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME",      str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from utterance_cache import UtteranceCache                      # noqa: E402
from vectorized_cache import VectorizedUtteranceCache, FP_DIM   # noqa: E402


VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train" / "waves"
VIVOS_TEST  = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"  / "waves"

DEFAULT_NS    = [32, 128, 512, 1024]
N_QUERIES     = 200
N_WARMUP      = 5
SR_TARGET     = 16000


def _gather_wavs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir(): continue
        out.extend(sorted(spk_dir.glob("*.wav")))
    return out


def _load_wav(p: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


def _jitter(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    noise_amp = 1e-3 * rng.uniform(0.5, 1.5)
    noise = (np.random.default_rng(rng.randrange(2**31))
             .normal(size=len(audio))
             .astype(np.float32)) * noise_amp
    return (audio + noise).astype(np.float32)


def _percentile(values_ns: list[int], q: float) -> float:
    if not values_ns:
        return 0.0
    return float(np.percentile(values_ns, q)) / 1e6


# ---------- Timed lookup primitives ----------

def _baseline_compare_ns(cache: UtteranceCache, fp, target_lang: str) -> tuple[int, Optional[int], float]:
    """Python loop comparison; returns (ns, chosen_id_or_None, best_score)."""
    t0 = time.perf_counter_ns()
    best_score = -1.0
    best_id: Optional[int] = None
    for k, entry in cache._store.items():
        if entry["target_lang"] != target_lang:
            continue
        score = float(np.dot(fp.vec, entry["fingerprint"].vec))
        if score > best_score:
            best_score = score
            best_id    = k
    t1 = time.perf_counter_ns()
    chosen = best_id if (best_id is not None and best_score >= cache.cosine_hit) else None
    return (t1 - t0), chosen, best_score


def _vectorized_compare_ns(cache: VectorizedUtteranceCache, fp, target_lang: str) -> tuple[int, Optional[int], float]:
    """Single matmul comparison; returns (ns, chosen_id_or_None, best_score)."""
    t0 = time.perf_counter_ns()
    n = len(cache._ids)
    chosen: Optional[int] = None
    best_score = -1.0
    if n > 0:
        scores = cache._fp_matrix[:n] @ fp.vec
        tl_arr = np.asarray(cache._target_langs)
        mask = (tl_arr == target_lang)
        if mask.any():
            masked = np.where(mask, scores, -np.inf)
            best_row = int(np.argmax(masked))
            best_score = float(masked[best_row])
            if best_score >= cache.cosine_hit:
                chosen = cache._ids[best_row]
    t1 = time.perf_counter_ns()
    return (t1 - t0), chosen, best_score


# ---------- Population (identical seed/order for both caches → identical contents) ----------

def _populate_both(base: UtteranceCache, vec: VectorizedUtteranceCache,
                   train_wavs: list[Path], target_n: int, rng: random.Random) -> None:
    base._store.clear(); base._next_id = 0
    vec._store.clear();  vec._next_id  = 0
    vec._fp_matrix.fill(0.0)
    vec._ids.clear()
    vec._target_langs.clear()
    # Pre-grow vec's matrix to target_n if needed
    if vec._fp_matrix.shape[0] < target_n:
        vec._fp_matrix = np.zeros((target_n, FP_DIM), dtype=np.float32)

    idx = 0
    base_n = len(train_wavs)
    while len(base._store) < target_n:
        wav_path = train_wavs[idx % base_n]
        audio, sr = _load_wav(wav_path)
        if idx >= base_n:
            audio = _jitter(audio, rng)
        fp = base.fingerprint(audio, sr)
        if fp.duration_s < base.min_duration_s:
            idx += 1
            if idx > target_n * 8:
                break
            continue
        # Bypass put() to avoid eviction; insert deterministically into both.
        base._next_id += 1
        eid = base._next_id
        base._store[eid] = {
            "fingerprint": fp, "target_lang": "en",
            "result": {"vi": f"u{idx}", "tgt": f"u{idx}"},
            "last_used": time.time(), "hits": 0, "inserted": time.time(),
        }
        # Mirror in vec cache with the same id and same fp object.
        vec._next_id = eid
        vec._store[eid] = {
            "fingerprint": fp, "target_lang": "en",
            "result": {"vi": f"u{idx}", "tgt": f"u{idx}"},
            "last_used": time.time(), "hits": 0, "inserted": time.time(),
        }
        row_i = len(vec._ids)
        if row_i >= vec._fp_matrix.shape[0]:
            grown = np.zeros((vec._fp_matrix.shape[0] * 2, FP_DIM), dtype=np.float32)
            grown[:row_i] = vec._fp_matrix[:row_i]
            vec._fp_matrix = grown
        vec._fp_matrix[row_i] = fp.vec
        vec._ids.append(eid)
        vec._target_langs.append("en")
        idx += 1
        if idx > target_n * 8:
            break


# ---------- One N cell ----------

def _bench_n(base: UtteranceCache, vec: VectorizedUtteranceCache,
             train_wavs: list[Path], test_wavs: list[Path],
             n: int, n_queries: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    print(f"  [populate] N={n} ... ", end="", flush=True)
    t_pop = time.perf_counter()
    _populate_both(base, vec, train_wavs, n, rng)
    print(f"done in {time.perf_counter() - t_pop:.1f}s "
          f"(base={len(base._store)} vec={len(vec._store)} matrix={vec.matrix_shape()})",
          flush=True)

    rng_q = random.Random(seed + 1)
    queries = [rng_q.choice(test_wavs) for _ in range(n_queries + N_WARMUP)]

    # Warmup
    for q in queries[:N_WARMUP]:
        audio, sr = _load_wav(q)
        fp = base.fingerprint(audio, sr)
        _baseline_compare_ns(base, fp, "en")
        _vectorized_compare_ns(vec, fp, "en")

    base_ns: list[int] = []
    vec_ns:  list[int] = []
    equiv_hits = 0          # both hit, same id, cosine within 1e-5
    equiv_misses = 0        # both miss
    disagreements = 0
    cosine_diffs: list[float] = []

    for q in queries[N_WARMUP:]:
        audio, sr = _load_wav(q)
        fp = base.fingerprint(audio, sr)
        b_ns, b_id, b_score = _baseline_compare_ns(base, fp, "en")
        v_ns, v_id, v_score = _vectorized_compare_ns(vec,  fp, "en")
        base_ns.append(b_ns)
        vec_ns.append(v_ns)

        if b_id == v_id:
            if b_id is None:
                equiv_misses += 1
            else:
                equiv_hits += 1
                cosine_diffs.append(abs(b_score - v_score))
        else:
            disagreements += 1
        # Always compare the unthresholded argmax cosine as a sanity (both compute it)
        cosine_diffs.append(abs(b_score - v_score))

    max_cos_diff = float(max(cosine_diffs)) if cosine_diffs else 0.0
    speedup_p50 = (_percentile(base_ns, 50) / _percentile(vec_ns, 50)) if _percentile(vec_ns, 50) > 0 else float("inf")
    speedup_p95 = (_percentile(base_ns, 95) / _percentile(vec_ns, 95)) if _percentile(vec_ns, 95) > 0 else float("inf")

    return {
        "n": n,
        "actual_entries": len(base._store),
        "n_queries": len(base_ns),
        "baseline": {
            "comparison_ms_mean": float(np.mean(base_ns)) / 1e6,
            "comparison_ms_p50":  _percentile(base_ns, 50),
            "comparison_ms_p95":  _percentile(base_ns, 95),
            "comparison_ms_p99":  _percentile(base_ns, 99),
        },
        "vectorized": {
            "comparison_ms_mean": float(np.mean(vec_ns)) / 1e6,
            "comparison_ms_p50":  _percentile(vec_ns, 50),
            "comparison_ms_p95":  _percentile(vec_ns, 95),
            "comparison_ms_p99":  _percentile(vec_ns, 99),
        },
        "speedup_p50": speedup_p50,
        "speedup_p95": speedup_p95,
        "equivalence": {
            "both_hit_same_id":   equiv_hits,
            "both_miss":          equiv_misses,
            "disagreements":      disagreements,
            "max_cosine_diff":    max_cos_diff,
            "within_1e-5":        bool(max_cos_diff < 1e-5),
        },
    }


# ---------- Driver ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns",        type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--n_queries", type=int, default=N_QUERIES)
    ap.add_argument("--out", default=str(PROJECT / "Results" / "vectorized_cache_bench.json"))
    args = ap.parse_args()

    train_wavs = _gather_wavs(VIVOS_TRAIN)
    test_wavs  = _gather_wavs(VIVOS_TEST)
    print(f"[data] VIVOS train wavs: {len(train_wavs)}")
    print(f"[data] VIVOS test  wavs: {len(test_wavs)}")
    if not train_wavs or not test_wavs:
        print("[error] VIVOS not found at expected path", file=sys.stderr)
        return 2

    max_n = max(args.ns)
    base = UtteranceCache(max_entries=max_n * 2, cosine_hit=0.95)
    vec  = VectorizedUtteranceCache(max_entries=max_n * 2, cosine_hit=0.95)

    results = {
        "config": {
            "ns":          args.ns,
            "n_queries":   args.n_queries,
            "n_warmup":    N_WARMUP,
            "vivos_train": len(train_wavs),
            "vivos_test":  len(test_wavs),
            "fp_dim":      FP_DIM,
        },
        "per_n": [],
    }

    print(f"\n=== Side-by-side: UtteranceCache vs VectorizedUtteranceCache ===")
    for n in args.ns:
        row = _bench_n(base, vec, train_wavs, test_wavs, n, args.n_queries)
        results["per_n"].append(row)

    # Pretty-print
    print("\n" + "=" * 116)
    print(f"{'N':>6}{'base.p50':>12}{'base.p95':>12}{'base.p99':>12}"
          f"{'vec.p50':>12}{'vec.p95':>12}{'vec.p99':>12}"
          f"{'spd.p50':>10}{'spd.p95':>10}{'equiv':>12}")
    print("-" * 116)
    for row in results["per_n"]:
        eq = row["equivalence"]
        ok = "OK" if (eq["disagreements"] == 0 and eq["within_1e-5"]) else "FAIL"
        print(f"{row['n']:>6}"
              f"{row['baseline']['comparison_ms_p50']:>12.4f}"
              f"{row['baseline']['comparison_ms_p95']:>12.4f}"
              f"{row['baseline']['comparison_ms_p99']:>12.4f}"
              f"{row['vectorized']['comparison_ms_p50']:>12.4f}"
              f"{row['vectorized']['comparison_ms_p95']:>12.4f}"
              f"{row['vectorized']['comparison_ms_p99']:>12.4f}"
              f"{row['speedup_p50']:>10.2f}x"
              f"{row['speedup_p95']:>10.2f}x"
              f"{ok:>12}")
    print("=" * 116)
    print("(all comparison-loop times in ms; speedup = baseline / vectorized; equiv checks max cosine diff < 1e-5 and 0 disagreements)")

    # Honest reporting
    print("\n[honest report]")
    for row in results["per_n"]:
        s = row["speedup_p50"]
        n = row["n"]
        verdict = (
            "BIG WIN" if s >= 10 else
            "win"     if s >= 5  else
            "marginal" if s >= 2 else
            "no meaningful speedup (Python overhead already low at this N)"
        )
        print(f"  N={n:>5}  speedup_p50={s:5.2f}x  -> {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
