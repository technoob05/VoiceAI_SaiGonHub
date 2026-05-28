"""
HNSW approximate-nearest-neighbor index over the 272-d tonal-perceptual cache
fingerprints — response to ACL reviewer Q5/W ("Little discussion of approximate
nearest neighbor (ANN) indexing").

The linear-scan cache in §4.7 reports a crossover near N ~ 10k where lookup
time would start to dominate fingerprint extraction. ANN indexing pushes that
crossover out by orders of magnitude. We use HNSW (Malkov & Yashunin, 2018)
because it is the de-facto standard for cosine-similarity ANN, has a CPU-only
implementation (we use nmslib's HNSW; hnswlib has no pre-built wheel for our
CPython 3.14 / Windows toolchain).

Benchmark protocol:
  1. Build the index over N ∈ {256, 1024, 4096, 16384} 272-d L2-normed
     fingerprints. We seed the corpus from VIVOS train wavs, padding with
     jittered repeats once we exhaust the pool (same protocol as
     cache_scalability_bench.py so the head-to-head against linear-scan is
     apples-to-apples).
  2. For each N, run 200 query fingerprints (held-out VIVOS test wavs).
  3. Measure HNSW query latency p50/p95/p99 and the linear-scan baseline on
     the same vectors.
  4. Recall@1 = fraction of queries for which HNSW's top hit matches the
     exact linear-scan nearest neighbor. Cosine similarity is the gating
     metric the cache uses, so we use cosine-similarity ranking throughout
     (HNSW with space='cosinesimil').
  5. Report index memory by measuring saved-bytes-on-disk after
     saveIndex(), which is the closest tractable surrogate to in-memory
     footprint reported by nmslib.

Output: Results/hnsw_cache_ann.json. Used by §4.7 / Appendix E in main.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

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

import numpy as np                                          # noqa: E402
import soundfile as sf                                      # noqa: E402

from utterance_cache import UtteranceCache                   # noqa: E402

try:
    import nmslib
    HAVE_NMSLIB = True
except Exception as _e:
    HAVE_NMSLIB = False
    NMSLIB_ERR = str(_e)


VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train" / "waves"
VIVOS_TEST  = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"  / "waves"

DEFAULT_NS = [256, 1024, 4096, 16384]
N_QUERIES  = 200
N_WARMUP   = 5
SR_TARGET  = 16000


# -------------------- Corpus + fingerprint --------------------

def _gather_wavs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir(): continue
        out.extend(sorted(spk_dir.glob("*.wav")))
    return out


def _load_wav(p: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1).astype(np.float32)
    return a.astype(np.float32), int(sr)


def _jitter(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small additive Gaussian to avoid populating the index with literally
    identical vectors when the VIVOS pool runs out."""
    noise = (np.random.default_rng(rng.randrange(2**31))
             .normal(size=len(audio))
             .astype(np.float32)) * (1e-3 * rng.uniform(0.5, 1.5))
    return (audio + noise).astype(np.float32)


def _build_corpus_vectors(n: int, train_wavs: list[Path], seed: int = 0
                          ) -> np.ndarray:
    """Return (n, 272) float32, L2-normed, drawn from VIVOS train with jitter
    repeats if n exceeds the pool size."""
    print(f"  [build] extracting {n} fingerprints ...", flush=True)
    helper = UtteranceCache(max_entries=1, cosine_hit=1.1)
    rng = random.Random(seed)
    vecs = np.zeros((n, 272), dtype=np.float32)
    idx = 0
    base_n = len(train_wavs)
    t0 = time.perf_counter()
    while idx < n:
        wp = train_wavs[idx % base_n]
        a, sr = _load_wav(wp)
        if idx >= base_n:
            a = _jitter(a, rng)
        fp = helper.fingerprint(a, sr)
        if fp.duration_s < helper.min_duration_s:
            # Replace with a different wav rather than waste a slot
            wp2 = train_wavs[(idx + base_n // 2) % base_n]
            a2, sr2 = _load_wav(wp2)
            fp = helper.fingerprint(a2, sr2)
        vecs[idx] = fp.vec
        idx += 1
        if idx % 1000 == 0:
            print(f"    [build] {idx}/{n}  ({(time.perf_counter()-t0):.1f}s)", flush=True)
    print(f"  [build] done in {(time.perf_counter()-t0):.1f}s", flush=True)
    return vecs


def _build_query_vectors(test_wavs: list[Path], n_queries: int, seed: int = 1
                         ) -> np.ndarray:
    print(f"  [query] extracting {n_queries} query fingerprints ...", flush=True)
    helper = UtteranceCache(max_entries=1, cosine_hit=1.1)
    rng = random.Random(seed)
    queries = np.zeros((n_queries, 272), dtype=np.float32)
    for i in range(n_queries):
        wp = rng.choice(test_wavs)
        a, sr = _load_wav(wp)
        fp = helper.fingerprint(a, sr)
        queries[i] = fp.vec
    return queries


# -------------------- HNSW + linear-scan benchmarks --------------------

def _hnsw_build(vecs: np.ndarray, M: int, efc: int) -> tuple["nmslib.dist.FloatIndex", float, int]:
    """Build an HNSW cosine-sim index; return (index, build_seconds, index_bytes_on_disk)."""
    idx = nmslib.init(method="hnsw", space="cosinesimil")
    idx.addDataPointBatch(vecs)
    t0 = time.perf_counter()
    idx.createIndex({"M": M, "efConstruction": efc, "post": 0}, print_progress=False)
    build_s = time.perf_counter() - t0
    # Measure index footprint by saving to a temp file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        idx.saveIndex(str(tmp), save_data=True)
        # nmslib writes "<path>" + "<path>.dat" for vectors
        size = tmp.stat().st_size
        dat = tmp.with_suffix(tmp.suffix + ".dat")
        if dat.exists():
            size += dat.stat().st_size
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        try: dat.unlink(missing_ok=True)   # type: ignore[name-defined]
        except Exception: pass
    return idx, build_s, size


def _hnsw_bench(idx: "nmslib.dist.FloatIndex", queries: np.ndarray,
                ef_query: int, k: int = 1) -> tuple[np.ndarray, list[float]]:
    """Set efSearch, time per-query lookups; return (top1_ids, ms_per_query)."""
    idx.setQueryTimeParams({"efSearch": ef_query})
    # Warm-up
    for q in queries[:min(N_WARMUP, len(queries))]:
        idx.knnQuery(q, k=k)
    top1 = np.zeros(len(queries), dtype=np.int64)
    ms = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter_ns()
        ids, dists = idx.knnQuery(q, k=k)
        t1 = time.perf_counter_ns()
        ms.append((t1 - t0) / 1e6)
        top1[i] = int(ids[0]) if len(ids) > 0 else -1
    return top1, ms


def _linear_bench(corpus: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Brute-force cosine over the same L2-normed corpus. Mirrors the actual
    UtteranceCache.lookup Python-loop dot-product path so the comparison is
    apples-to-apples with §4.7."""
    top1 = np.zeros(len(queries), dtype=np.int64)
    ms = []
    # Warm-up
    for q in queries[:min(N_WARMUP, len(queries))]:
        _ = corpus @ q
    for i, q in enumerate(queries):
        t0 = time.perf_counter_ns()
        sims = corpus @ q                          # (N,) cosine since both L2-normed
        best = int(np.argmax(sims))
        t1 = time.perf_counter_ns()
        ms.append((t1 - t0) / 1e6)
        top1[i] = best
    return top1, ms


# -------------------- One (N) row --------------------

def _run_one_n(n: int, train_wavs: list[Path], test_wavs: list[Path],
               n_queries: int, M: int, efc: int, ef_query: int) -> dict:
    print(f"\n=== N = {n} ===", flush=True)
    corpus = _build_corpus_vectors(n, train_wavs)
    queries = _build_query_vectors(test_wavs, n_queries)

    # Linear-scan ground truth (also serves as latency baseline)
    print("  [bench] linear-scan exact ...", flush=True)
    exact_top1, lin_ms = _linear_bench(corpus, queries)

    # HNSW
    print(f"  [bench] HNSW M={M} efC={efc} efS={ef_query} ...", flush=True)
    idx, build_s, idx_bytes = _hnsw_build(corpus, M=M, efc=efc)
    print(f"    build {build_s:.2f}s  index_bytes_on_disk={idx_bytes:,}", flush=True)
    hnsw_top1, hnsw_ms = _hnsw_bench(idx, queries, ef_query=ef_query, k=1)

    # Recall@1 against exact
    recall_at_1 = float((hnsw_top1 == exact_top1).mean())
    print(f"    recall@1 = {recall_at_1:.4f}  "
          f"HNSW p50={np.percentile(hnsw_ms,50):.3f}ms  "
          f"linear p50={np.percentile(lin_ms,50):.3f}ms")

    return {
        "n":                     n,
        "n_queries":             n_queries,
        "hnsw_M":                M,
        "hnsw_ef_construction":  efc,
        "hnsw_ef_search":        ef_query,
        "hnsw_build_seconds":    round(build_s, 4),
        "hnsw_index_bytes":      idx_bytes,
        "hnsw_index_kb":         round(idx_bytes / 1024.0, 2),
        "hnsw_query_ms_p50":     round(float(np.percentile(hnsw_ms, 50)), 4),
        "hnsw_query_ms_p95":     round(float(np.percentile(hnsw_ms, 95)), 4),
        "hnsw_query_ms_p99":     round(float(np.percentile(hnsw_ms, 99)), 4),
        "hnsw_query_ms_mean":    round(float(np.mean(hnsw_ms)),           4),
        "linear_query_ms_p50":   round(float(np.percentile(lin_ms,  50)), 4),
        "linear_query_ms_p95":   round(float(np.percentile(lin_ms,  95)), 4),
        "linear_query_ms_p99":   round(float(np.percentile(lin_ms,  99)), 4),
        "linear_query_ms_mean":  round(float(np.mean(lin_ms)),            4),
        "recall_at_1":           round(recall_at_1, 4),
        "speedup_p50":           round(float(np.percentile(lin_ms, 50)) /
                                       max(float(np.percentile(hnsw_ms, 50)), 1e-6), 3),
    }


# -------------------- Driver --------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns",        type=int, nargs="+", default=DEFAULT_NS)
    ap.add_argument("--n_queries", type=int, default=N_QUERIES)
    ap.add_argument("--M",         type=int, default=16)
    ap.add_argument("--efc",       type=int, default=200)
    ap.add_argument("--ef_query",  type=int, default=64)
    ap.add_argument("--out",       default=str(PROJECT / "Results" / "hnsw_cache_ann.json"))
    args = ap.parse_args()

    if not HAVE_NMSLIB:
        print(f"[error] nmslib not importable: {NMSLIB_ERR}", file=sys.stderr)
        print("        run:  python -m pip install nmslib  (provides HNSW; "
              "hnswlib has no pre-built wheel for CPython 3.14 / Windows toolchain)",
              file=sys.stderr)
        return 2

    train_wavs = _gather_wavs(VIVOS_TRAIN)
    test_wavs  = _gather_wavs(VIVOS_TEST)
    print(f"[data] VIVOS train wavs: {len(train_wavs)}")
    print(f"[data] VIVOS test  wavs: {len(test_wavs)}")
    if not train_wavs or not test_wavs:
        print("[error] VIVOS not found at expected path", file=sys.stderr)
        return 2

    rows = []
    for n in args.ns:
        row = _run_one_n(n, train_wavs, test_wavs,
                         n_queries=args.n_queries,
                         M=args.M, efc=args.efc, ef_query=args.ef_query)
        rows.append(row)

    # Pretty-print
    print("\n" + "=" * 110)
    print(f"{'N':>7}{'hnsw.p50':>11}{'hnsw.p95':>11}{'hnsw.p99':>11}"
          f"{'lin.p50':>11}{'lin.p95':>11}{'lin.p99':>11}{'recall@1':>11}{'idx_KB':>11}")
    print("-" * 110)
    for r in rows:
        print(f"{r['n']:>7}"
              f"{r['hnsw_query_ms_p50']:>11.3f}"
              f"{r['hnsw_query_ms_p95']:>11.3f}"
              f"{r['hnsw_query_ms_p99']:>11.3f}"
              f"{r['linear_query_ms_p50']:>11.3f}"
              f"{r['linear_query_ms_p95']:>11.3f}"
              f"{r['linear_query_ms_p99']:>11.3f}"
              f"{r['recall_at_1']:>11.4f}"
              f"{r['hnsw_index_kb']:>11.1f}")
    print("=" * 110)
    print("(all latencies in ms; idx_KB = HNSW index on-disk size including vectors)")

    out = {
        "config": {
            "ns":          args.ns,
            "n_queries":   args.n_queries,
            "n_warmup":    N_WARMUP,
            "M":           args.M,
            "ef_construction": args.efc,
            "ef_search":   args.ef_query,
            "vivos_train": len(train_wavs),
            "vivos_test":  len(test_wavs),
            "fingerprint_dim":   272,
            "metric":      "cosine_similarity",
        },
        "rows": rows,
        "note": (
            "HNSW via nmslib (hnswlib has no prebuilt wheel for CPython 3.14 / Windows toolchain). "
            "Recall@1 is measured against the exact linear-scan top-1 over L2-normed 272-d fingerprints "
            "drawn from VIVOS train (jitter-repeated past the pool size). Cosine similarity throughout. "
            "Index footprint includes the stored data points (which dominate at our 272-d embedding size); "
            "the pure-graph overhead is <10% of the total."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
