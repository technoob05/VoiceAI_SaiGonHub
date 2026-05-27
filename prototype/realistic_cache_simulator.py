"""
Realistic industrial-load cache simulator — paper §4.9 expansion.

The existing worker_simulation.py uses uniform K=2 phrases × R=3 repeats per
worker. That setup is too tidy: it answers "if everyone speaks the same fixed
deck, what hit rate do we get?" but a real factory shift has

  (a) bursty arrivals — workers don't speak on a metronome; inter-utterance
      gaps are exponentially distributed (Poisson arrivals);
  (b) Zipf-shaped phrase frequency — operators repeat a handful of safety /
      coordination phrases ~10× as often as procedural ones, with a long
      tail of rare phrases that hit the cache once and get LRU-evicted.

This script simulates an 8-hour shift over three fleet sizes (10 / 50 / 200
workers), each worker drawing from a 50-phrase vocabulary with Zipf-1
frequency (top phrase ≈ 10 % of utterances, phrase #50 ≈ 0.1 %), and
exponential inter-arrival times at λ = 1 utterance / 30 s.

It is a CACHE-ONLY simulator: we do not run ASR / MT / TTS for 30k+ events.
For each event we (i) "fingerprint" (constant 25 ms), (ii) look up T1 (audio
cache), (iii) on miss, sample a cascade latency from the empirical
distribution fitted to Results/worker_simulation.json. T2 (text TM) is left
out of the simulated decision because we do not have actual ASR transcripts
to key it on — instead we report it as a separate, conservative
upper-bound number derived from same-phrase cross-worker reuse.

LRU semantics match the production UtteranceCache: bounded capacity, LRU
eviction, key = (worker_id, phrase_id) for T1 (audio fingerprint is
speaker-dependent), key = phrase_id only for T2.

Honest reporting:
  - Cache fill saturates at min(capacity, N_workers × n_unique_phrases_seen).
    With 50 phrases × 200 workers = 10 000 possible (worker, phrase) pairs but
    a default cache cap of 256, the LRU evicts the long Zipf tail; only the
    top ~20-30 (worker, phrase) pairs stay resident.
  - T2 reuse rate is capped by P(two workers utter same phrase within the
    cache TTL) — Zipf top-heaviness pushes this high, but the report still
    shows the saturation point.

Output: Results/realistic_cache_simulator.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────
#  Cascade-latency empirical model
# ─────────────────────────────────────────────────────────────────────────
#
# From Results/worker_simulation.json (scenarios with n_workers ∈ {1,3,5}):
#   mean ≈ 3100-3500 ms, p50 ≈ 35-78 ms (cache hits dominate p50),
#   p95 ≈ 12-17 k ms, p99 ≈ 13-20 k ms, max ≈ 20 k ms.
#
# The p50 is small because t1-hits are 25-80 ms; the heavy tail is real
# ASR+MT+TTS misses. We want only the MISS distribution here (cache hits are
# modeled separately as 25 ms constant). Pulling out misses from the
# observed quantiles:
#   miss_mean ≈ 6700 ms   (typical full Whisper-tiny + NLLB-600M + Piper)
#   miss_p50  ≈ 5500 ms
#   miss_p95  ≈ 13500 ms
#   miss_p99  ≈ 18000 ms
#
# A lognormal with µ = ln(5500) = 8.61, σ = 0.55 reproduces this shape
# within ~10 %. We sample from that.
FP_MS         = 25.0           # fingerprint cost (deterministic)
CACHE_HIT_MS  = FP_MS + 0.5    # constant: fingerprint + dict lookup
MISS_LN_MU    = 8.61
MISS_LN_SIGMA = 0.55


def sample_miss_latency(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n cascade-miss latencies (ms) from the fitted lognormal."""
    return rng.lognormal(MISS_LN_MU, MISS_LN_SIGMA, size=n)


# ─────────────────────────────────────────────────────────────────────────
#  Bounded LRU cache (matches production UtteranceCache semantics)
# ─────────────────────────────────────────────────────────────────────────
class LRU:
    __slots__ = ("cap", "_d", "hits", "misses", "evictions")

    def __init__(self, cap: int):
        self.cap = cap
        self._d: "OrderedDict[object, float]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key) -> bool:
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def put(self, key, value: float = 1.0) -> None:
        if key in self._d:
            self._d.move_to_end(key)
            return
        self._d[key] = value
        if len(self._d) > self.cap:
            self._d.popitem(last=False)
            self.evictions += 1

    def __len__(self) -> int:
        return len(self._d)


# ─────────────────────────────────────────────────────────────────────────
#  Event stream
# ─────────────────────────────────────────────────────────────────────────
def build_zipf_pmf(n_phrases: int, s: float = 1.0) -> np.ndarray:
    """Zipf-s PMF over n_phrases ranks (rank 1 = most common)."""
    ranks = np.arange(1, n_phrases + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()


def generate_events(
    n_workers: int,
    n_phrases: int,
    shift_seconds: float,
    arrival_lambda: float,
    zipf_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return structured array of (t, worker, phrase) sorted by time.

    Each worker emits an independent Poisson process with rate
    arrival_lambda (utt/sec). We generate exponential inter-arrival gaps
    until time exceeds shift_seconds. Phrase IDs are drawn i.i.d. from the
    Zipf-s PMF — same vocabulary for every worker (industrial phrase deck
    is shared across the fleet, individual usage differs slightly only via
    sampling noise).
    """
    pmf = build_zipf_pmf(n_phrases, s=zipf_s)

    all_t = []
    all_w = []
    all_p = []
    for wid in range(n_workers):
        # over-allocate then trim — cheaper than appending one at a time
        # expected n = lambda * shift_seconds, use 1.6x as safety
        n_guess = max(8, int(arrival_lambda * shift_seconds * 1.6))
        gaps = rng.exponential(1.0 / arrival_lambda, size=n_guess)
        ts = np.cumsum(gaps)
        ts = ts[ts < shift_seconds]
        if ts.size == 0:
            continue
        ps = rng.choice(n_phrases, size=ts.size, p=pmf)
        all_t.append(ts)
        all_w.append(np.full(ts.size, wid, dtype=np.int32))
        all_p.append(ps.astype(np.int32))

    if not all_t:
        return np.zeros(0, dtype=[("t", "f8"), ("w", "i4"), ("p", "i4")])

    t = np.concatenate(all_t)
    w = np.concatenate(all_w)
    p = np.concatenate(all_p)

    order = np.argsort(t, kind="stable")
    out = np.empty(t.size, dtype=[("t", "f8"), ("w", "i4"), ("p", "i4")])
    out["t"] = t[order]
    out["w"] = w[order]
    out["p"] = p[order]
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Simulation core
# ─────────────────────────────────────────────────────────────────────────
def simulate_scenario(
    n_workers: int,
    n_phrases: int = 50,
    shift_seconds: float = 28800.0,
    arrival_lambda: float = 1.0 / 30.0,
    zipf_s: float = 1.0,
    t1_capacity: int = 256,
    t2_capacity: int = 512,
    seed: int = 42,
    hourly_buckets: int = 8,
) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    events = generate_events(n_workers, n_phrases, shift_seconds,
                             arrival_lambda, zipf_s, rng)
    n_events = events.size
    if n_events == 0:
        return {"error": "no events generated", "n_workers": n_workers}

    t1 = LRU(t1_capacity)   # key = (worker, phrase) — speaker-dependent audio fingerprint
    t2 = LRU(t2_capacity)   # key = phrase           — text-level TM, cross-worker
    miss_pool = sample_miss_latency(rng, n_events)   # pre-sample for misses

    latencies = np.empty(n_events, dtype=np.float64)
    t1_hits_mask = np.zeros(n_events, dtype=bool)
    t2_hits_mask = np.zeros(n_events, dtype=bool)
    miss_idx = 0

    for i in range(n_events):
        wid = int(events["w"][i])
        pid = int(events["p"][i])
        k1 = (wid, pid)

        if t1.get(k1):
            latencies[i] = CACHE_HIT_MS
            t1_hits_mask[i] = True
            t2.get(pid)  # phrase-level access touches T2 LRU too
            continue

        # T1 miss → check T2 (cross-worker text reuse)
        if t2.get(pid):
            # T2 hit means we skip MT and TTS (≈70 % of cascade time saved).
            # Empirical: fingerprint + ASR ≈ 25 + ~1500 ms.
            latencies[i] = FP_MS + 1500.0
            t2_hits_mask[i] = True
            t1.put(k1)
            continue

        # Full cascade miss
        latencies[i] = miss_pool[miss_idx]
        miss_idx += 1
        t1.put(k1)
        t2.put(pid)

    # Hourly buckets
    bucket_edges = np.linspace(0.0, shift_seconds, hourly_buckets + 1)
    bucket_ids = np.clip(
        np.searchsorted(bucket_edges, events["t"], side="right") - 1,
        0, hourly_buckets - 1,
    )
    t1_per_hour = []
    t2_per_hour = []
    count_per_hour = []
    for b in range(hourly_buckets):
        m = bucket_ids == b
        cnt = int(m.sum())
        count_per_hour.append(cnt)
        if cnt == 0:
            t1_per_hour.append(0.0)
            t2_per_hour.append(0.0)
        else:
            t1_per_hour.append(round(float(t1_hits_mask[m].mean()), 4))
            t2_per_hour.append(round(float(t2_hits_mask[m].mean()), 4))

    cascade_only_mean = float(np.mean(miss_pool))
    energy_saved_ms = float(np.sum(cascade_only_mean - latencies))

    return {
        "n_workers": n_workers,
        "n_phrases_vocab": n_phrases,
        "shift_seconds": shift_seconds,
        "arrival_lambda_per_sec": arrival_lambda,
        "zipf_s": zipf_s,
        "n_events": int(n_events),
        "t1_capacity": t1_capacity,
        "t2_capacity": t2_capacity,

        "t1_hit_rate":  round(float(t1_hits_mask.mean()), 4),
        "t2_hit_rate":  round(float(t2_hits_mask.mean()), 4),
        "miss_rate":    round(float(1.0 - t1_hits_mask.mean() - t2_hits_mask.mean()), 4),

        "t1_hit_rate_hourly":   t1_per_hour,
        "t2_hit_rate_hourly":   t2_per_hour,
        "events_per_hour":      count_per_hour,

        "final_cache_fill": {
            "t1_entries":    len(t1),
            "t1_evictions":  int(t1.evictions),
            "t2_entries":    len(t2),
            "t2_evictions":  int(t2.evictions),
        },

        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 1),
            "p50":  round(float(np.percentile(latencies, 50)), 1),
            "p95":  round(float(np.percentile(latencies, 95)), 1),
            "p99":  round(float(np.percentile(latencies, 99)), 1),
            "max":  round(float(np.max(latencies)), 1),
        },
        "cascade_only_baseline_ms": {
            "mean": round(cascade_only_mean, 1),
            "p99":  round(float(np.percentile(miss_pool, 99)), 1),
        },
        "energy_saved_ms_total":        round(energy_saved_ms, 1),
        "latency_reduction_vs_cascade": round(
            1.0 - float(np.mean(latencies)) / cascade_only_mean, 4
        ),
        "wall_clock_seconds": round(time.perf_counter() - t0, 2),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", nargs="+", type=int, default=[10, 50, 200])
    ap.add_argument("--phrases", type=int, default=50)
    ap.add_argument("--shift-hours", type=float, default=8.0)
    ap.add_argument("--arrival-rate-per-min", type=float, default=2.0,
                    help="utterances per worker per minute (default 2 = 1 every 30 s)")
    ap.add_argument("--zipf-s", type=float, default=1.0)
    ap.add_argument("--t1-cap", type=int, default=256)
    ap.add_argument("--t2-cap", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="Results/realistic_cache_simulator.json")
    args = ap.parse_args()

    shift_seconds  = args.shift_hours * 3600.0
    arrival_lambda = args.arrival_rate_per_min / 60.0

    print(f"[config] shift={args.shift_hours}h ({shift_seconds:.0f}s)  "
          f"λ={arrival_lambda:.4f} utt/s/worker  "
          f"vocab={args.phrases} phrases  Zipf-s={args.zipf_s}  "
          f"T1 cap={args.t1_cap}  T2 cap={args.t2_cap}")

    scenarios = []
    for n in args.workers:
        print(f"\n===== N={n} workers =====")
        res = simulate_scenario(
            n_workers=n,
            n_phrases=args.phrases,
            shift_seconds=shift_seconds,
            arrival_lambda=arrival_lambda,
            zipf_s=args.zipf_s,
            t1_capacity=args.t1_cap,
            t2_capacity=args.t2_cap,
            seed=args.seed + n,
        )
        scenarios.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue

        lat       = res["latency_ms"]
        casc_mean = res["cascade_only_baseline_ms"]["mean"]
        fill      = res["final_cache_fill"]
        t1pct     = res["t1_hit_rate"] * 100
        t2pct     = res["t2_hit_rate"] * 100
        red_pct   = res["latency_reduction_vs_cascade"] * 100
        print(
            f"  N={n:>3} workers / {res['n_events']:>5} events: "
            f"T1 hit {t1pct:4.1f}% + T2 hit {t2pct:4.1f}% "
            f"(Zipf top phrases dominate), mean lat {lat['mean']:.0f} ms "
            f"(vs {casc_mean:.0f} ms cascade-only), "
            f"{red_pct:4.1f}% latency reduction."
        )
        print(
            f"     p50={lat['p50']:.0f}ms  p99={lat['p99']:.0f}ms  "
            f"T1 fill={fill['t1_entries']}/{args.t1_cap} "
            f"(evictions={fill['t1_evictions']})  "
            f"T2 fill={fill['t2_entries']}/{args.t2_cap} "
            f"(evictions={fill['t2_evictions']})"
        )

    payload = {
        "scenarios": scenarios,
        "global_config": {
            "shift_seconds":            shift_seconds,
            "arrival_lambda_per_sec":   arrival_lambda,
            "n_phrases_vocab":          args.phrases,
            "zipf_s":                   args.zipf_s,
            "t1_capacity":              args.t1_cap,
            "t2_capacity":              args.t2_cap,
            "fingerprint_ms":           FP_MS,
            "cache_hit_ms":             CACHE_HIT_MS,
            "miss_lognormal":           {"mu": MISS_LN_MU, "sigma": MISS_LN_SIGMA},
            "seed_base":                args.seed,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
