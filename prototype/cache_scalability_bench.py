"""
Cache scalability micro-benchmark — lookup latency vs cache size N.

Question this script answers: at what N does the per-entry comparison loop
(numpy dot products in a Python `for` over OrderedDict) start to dominate
the fingerprint-compute cost (~20 ms on ~4 s VIVOS audio)? And how does the
two-stream voting variant — two independent dots + AND-gate — scale next
to the single-vector UtteranceCache?

Both caches are imported READ-ONLY; we never modify production code.

Method:
  - Populate the cache with N real VIVOS train fingerprints (repeat with
    small jitter when we exceed the ~1380-utterance VIVOS train pool).
  - For each N, run 200 lookup queries from VIVOS test wavs.
  - Time fingerprint compute + comparison loop separately via
    perf_counter_ns; report p50/p95/p99.
  - 5 warm-up queries before each measurement block.

Output:
  - Results/cache_scalability_bench.json (per-N table for both caches)
  - Console pretty-print table.
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
from two_stream_voting import TwoStreamCache, TwoStreamConfig   # noqa: E402


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
    """Slight perturbation so we can pad the train pool past 1380 entries
    without storing literally identical fingerprints (which would collapse
    the comparison loop in a non-realistic way)."""
    noise_amp = 1e-3 * rng.uniform(0.5, 1.5)
    noise = (np.random.default_rng(rng.randrange(2**31))
             .normal(size=len(audio))
             .astype(np.float32)) * noise_amp
    return (audio + noise).astype(np.float32)


def _percentile(values_ns: list[int], q: float) -> float:
    """Return percentile in milliseconds."""
    if not values_ns:
        return 0.0
    return float(np.percentile(values_ns, q)) / 1e6


# ---------- Lookup wrappers that decouple fingerprint vs comparison timing ----------

def _utterance_lookup_timed(cache: UtteranceCache, fp, target_lang: str) -> tuple[int, int]:
    """Return (fingerprint_ns_already_consumed=0, comparison_ns).
    We pre-compute fp outside; here we only time the comparison loop."""
    t0 = time.perf_counter_ns()
    # Replicate the comparison-loop body without touching internal stats:
    best_score = -1.0
    best_id: Optional[int] = None
    for k, entry in cache._store.items():
        if entry["target_lang"] != target_lang:
            continue
        score = float(np.dot(fp.vec, entry["fingerprint"].vec))
        if score > best_score:
            best_score = score
            best_id    = k
    if best_id is not None and best_score >= cache.cosine_hit:
        # Don't mutate LRU here — keep timing pure
        pass
    t1 = time.perf_counter_ns()
    return (t1 - t0)


def _twostream_lookup_timed(cache: TwoStreamCache, fp, target_lang: str) -> int:
    t0 = time.perf_counter_ns()
    best_tone_id = best_seg_id = None
    best_tone = best_seg = -1.0
    for k, e in cache._store.items():
        if e.get("target_lang") != target_lang:
            continue
        t_score = float(np.dot(fp.tone,    e["fp"].tone))
        s_score = float(np.dot(fp.segment, e["fp"].segment))
        if t_score > best_tone:
            best_tone, best_tone_id = t_score, k
        if s_score > best_seg:
            best_seg, best_seg_id = s_score, k
    # AND gate (just to mirror real path's branch cost)
    _ = (best_tone_id is not None and
         best_tone >= cache.cfg.tone_threshold and
         best_seg  >= cache.cfg.segment_threshold and
         best_tone_id == best_seg_id)
    t1 = time.perf_counter_ns()
    return (t1 - t0)


# ---------- Population ----------

def _populate_utterance(cache: UtteranceCache, train_wavs: list[Path],
                        target_n: int, rng: random.Random) -> None:
    """Fill cache up to target_n entries (with jitter-repeats beyond the pool)."""
    cache._store.clear()
    cache._next_id = 0
    idx = 0
    base_n = len(train_wavs)
    while len(cache._store) < target_n:
        wav_path = train_wavs[idx % base_n]
        audio, sr = _load_wav(wav_path)
        if idx >= base_n:
            audio = _jitter(audio, rng)
        fp = cache.fingerprint(audio, sr)
        if fp.duration_s < cache.min_duration_s:
            idx += 1
            continue
        # bypass put() so we don't trigger eviction — we want exactly target_n
        cache._next_id += 1
        cache._store[cache._next_id] = {
            "fingerprint": fp,
            "target_lang": "en",
            "result":      {"vi": f"u{idx}", "tgt": f"u{idx}"},
            "last_used":   time.time(),
            "hits":        0,
            "inserted":    time.time(),
        }
        idx += 1
        if idx > target_n * 4:  # safety
            break


def _populate_twostream(cache: TwoStreamCache, train_wavs: list[Path],
                        target_n: int, rng: random.Random) -> None:
    cache._store.clear()
    cache._next_id = 0
    idx = 0
    base_n = len(train_wavs)
    while len(cache._store) < target_n:
        wav_path = train_wavs[idx % base_n]
        audio, sr = _load_wav(wav_path)
        if idx >= base_n:
            audio = _jitter(audio, rng)
        fp = cache.fingerprint(audio, sr)
        cache._next_id += 1
        cache._store[cache._next_id] = {
            "fp":          fp,
            "target_lang": "en",
            "result":      {"vi": f"u{idx}", "tgt": f"u{idx}"},
            "inserted":    time.time(),
        }
        idx += 1
        if idx > target_n * 4:
            break


# ---------- One benchmark cell ----------

def _bench_one(cache_name: str, build_cache, lookup_timed, populate_fn,
               train_wavs: list[Path], test_wavs: list[Path],
               n: int, n_queries: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    cache = build_cache(n)
    print(f"  [populate] {cache_name} N={n} ... ", end="", flush=True)
    t_pop = time.perf_counter()
    populate_fn(cache, train_wavs, n, rng)
    print(f"done in {time.perf_counter() - t_pop:.1f}s "
          f"(actual entries: {len(cache._store)})", flush=True)

    # Pick query wavs
    rng_q = random.Random(seed + 1)
    queries = [rng_q.choice(test_wavs) for _ in range(n_queries + N_WARMUP)]

    # Warm-up
    for q in queries[:N_WARMUP]:
        audio, sr = _load_wav(q)
        fp = cache.fingerprint(audio, sr)
        _ = lookup_timed(cache, fp, "en")

    fp_ns_list:   list[int] = []
    cmp_ns_list:  list[int] = []
    total_ns_list: list[int] = []

    for q in queries[N_WARMUP:]:
        audio, sr = _load_wav(q)
        t_fp0 = time.perf_counter_ns()
        fp = cache.fingerprint(audio, sr)
        t_fp1 = time.perf_counter_ns()
        cmp_ns = lookup_timed(cache, fp, "en")
        fp_ns_list.append(t_fp1 - t_fp0)
        cmp_ns_list.append(cmp_ns)
        total_ns_list.append((t_fp1 - t_fp0) + cmp_ns)

    return {
        "n": n,
        "actual_entries": len(cache._store),
        "n_queries": len(total_ns_list),
        "fingerprint_ms_mean":  float(np.mean(fp_ns_list))  / 1e6,
        "fingerprint_ms_p50":   _percentile(fp_ns_list,  50),
        "fingerprint_ms_p95":   _percentile(fp_ns_list,  95),
        "fingerprint_ms_p99":   _percentile(fp_ns_list,  99),
        "comparison_ms_mean":   float(np.mean(cmp_ns_list)) / 1e6,
        "comparison_ms_p50":    _percentile(cmp_ns_list, 50),
        "comparison_ms_p95":    _percentile(cmp_ns_list, 95),
        "comparison_ms_p99":    _percentile(cmp_ns_list, 99),
        "total_ms_p50":         _percentile(total_ns_list, 50),
        "total_ms_p95":         _percentile(total_ns_list, 95),
        "total_ms_p99":         _percentile(total_ns_list, 99),
    }


# ---------- Driver ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns",        type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--n_queries", type=int, default=N_QUERIES)
    ap.add_argument("--out", default=str(PROJECT / "Results" / "cache_scalability_bench.json"))
    args = ap.parse_args()

    train_wavs = _gather_wavs(VIVOS_TRAIN)
    test_wavs  = _gather_wavs(VIVOS_TEST)
    print(f"[data] VIVOS train wavs: {len(train_wavs)}")
    print(f"[data] VIVOS test  wavs: {len(test_wavs)}")
    if not train_wavs or not test_wavs:
        print("[error] VIVOS not found at expected path", file=sys.stderr)
        return 2

    max_n = max(args.ns)

    def _build_utterance(n: int) -> UtteranceCache:
        return UtteranceCache(max_entries=max(n, max_n) * 2, cosine_hit=0.95)

    def _build_twostream(n: int) -> TwoStreamCache:
        return TwoStreamCache(TwoStreamConfig(max_entries=max(n, max_n) * 2))

    results = {
        "config": {
            "ns":          args.ns,
            "n_queries":   args.n_queries,
            "n_warmup":    N_WARMUP,
            "vivos_train": len(train_wavs),
            "vivos_test":  len(test_wavs),
        },
        "utterance_cache": [],
        "two_stream_cache": [],
    }

    print(f"\n=== UtteranceCache (single 272-d) ===")
    for n in args.ns:
        row = _bench_one("UtteranceCache", _build_utterance,
                         _utterance_lookup_timed, _populate_utterance,
                         train_wavs, test_wavs, n, args.n_queries)
        results["utterance_cache"].append(row)

    print(f"\n=== TwoStreamCache (6-d + 260-d voting) ===")
    for n in args.ns:
        row = _bench_one("TwoStreamCache", _build_twostream,
                         _twostream_lookup_timed, _populate_twostream,
                         train_wavs, test_wavs, n, args.n_queries)
        results["two_stream_cache"].append(row)

    # Pretty-print
    print("\n" + "=" * 110)
    print(f"{'cache':<18}{'N':>6}{'fp.p50':>10}{'fp.p95':>10}{'fp.p99':>10}"
          f"{'cmp.p50':>10}{'cmp.p95':>10}{'cmp.p99':>10}"
          f"{'tot.p50':>10}{'tot.p95':>10}{'tot.p99':>10}")
    print("-" * 110)
    for name, key in [("UtteranceCache", "utterance_cache"),
                      ("TwoStreamCache", "two_stream_cache")]:
        for row in results[key]:
            print(f"{name:<18}{row['n']:>6}"
                  f"{row['fingerprint_ms_p50']:>10.2f}"
                  f"{row['fingerprint_ms_p95']:>10.2f}"
                  f"{row['fingerprint_ms_p99']:>10.2f}"
                  f"{row['comparison_ms_p50']:>10.3f}"
                  f"{row['comparison_ms_p95']:>10.3f}"
                  f"{row['comparison_ms_p99']:>10.3f}"
                  f"{row['total_ms_p50']:>10.2f}"
                  f"{row['total_ms_p95']:>10.2f}"
                  f"{row['total_ms_p99']:>10.2f}")
    print("=" * 110)
    print("(fp = fingerprint compute; cmp = comparison loop; tot = fp + cmp; all in ms)")

    # Crossover analysis
    print("\n[analysis] comparison-vs-fingerprint ratio per N:")
    for name, key in [("UtteranceCache", "utterance_cache"),
                      ("TwoStreamCache", "two_stream_cache")]:
        for row in results[key]:
            ratio = row["comparison_ms_mean"] / max(row["fingerprint_ms_mean"], 1e-9)
            tag = " <-- comparison dominates" if ratio >= 1.0 else ""
            print(f"  {name:<18} N={row['n']:>5}  "
                  f"cmp/fp = {ratio*100:6.2f}%{tag}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
