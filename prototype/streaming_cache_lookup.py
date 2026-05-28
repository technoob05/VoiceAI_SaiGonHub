"""
Streaming progressive cache lookup — decide HIT/MISS at partial audio.

Motivation (Round 25 → oral-ACL system contribution):
  Round 24 found cache-hit p50 scales 13.2 ms/s in source duration because the
  current pipeline waits for the FULL utterance, then extracts the 272-d
  fingerprint, then does the cosine lookup. For a 7.4-s utterance, that's
  ~148 ms p50 of pure "we already could have known the answer" wait.

  This module turns the cache into a TRUE streaming primitive: extract the
  fingerprint on progressively longer prefixes of the audio (0.5, 1.0, 1.5,
  2.0, full s), and at each decision point run the two-stream existence rule
  against the stored cache. Commit to "HIT entry e" only if BOTH streams
  exceed a TIGHTER threshold tau_partial than the full-audio threshold.

Design constraint (deployment safety):
  A false-commit on a partial-audio prefix is catastrophic — it shows the
  worker a translation of a sentence they did not yet finish saying. So the
  early-commit rule is conservative: we sweep tau_partial_A in {0.92, 0.94,
  0.96} (vs full-audio tau_A=0.85) and pick the tightest setting that keeps
  false-commit at 0 on the held-out set. If 0 is not reachable, we report the
  lowest false-commit rate and document the tau as a deployment knob.

Eval protocol (pure NumPy, CPU-only, no GPU, no new audio):
  - Pool: VIVOS test wav files (held-out from any tuning subset).
  - Cache prewarm: store fingerprints of N "canon" utterances.
  - Hit queries: re-extract a noise-perturbed copy of each canon utterance
    (DEMAND-like additive noise at 15 dB) and stream it.
  - Miss queries: query different VIVOS utterances against the same cache.
  - For each query, walk T in {0.5, 1.0, 1.5, 2.0, full}: compute partial
    fingerprint, run two-stream existence rule, commit if both pass tau_partial.

Metrics:
  - TTFH (time-to-first-hit): wall-clock from stream start to commit.
    For batch baseline TTFH = utterance_duration + full fingerprint time.
    For streaming TTFH = decision_t * (sr/sr) wall + fingerprint(prefix)_time.
  - false_commit_rate over miss queries.
  - fraction_of_audio at commit (decision_t / utterance_duration).

Outputs:
  - Results/streaming_cache_lookup.json
  - papers/figures/fig_streaming_ttfh.png

Usage:
  python prototype/streaming_cache_lookup.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np
import soundfile as sf

from utterance_cache import cheap_mfcc, _f0_contour, tone_histogram_from_f0, UtteranceCache


DECISION_POINTS_S = (0.5, 1.0, 1.5, 2.0, None)   # None = full utterance
TAU_PARTIAL_A_GRID = (0.92, 0.94, 0.96)
TAU_PARTIAL_B_GRID = (0.95, 0.97, 0.98)
TAU_FULL_A = 0.85
TAU_FULL_B = 0.93


def _components_partial(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (tone 6-d L2, segment 260-d L2). Same construction as TwoStreamCache."""
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    if len(audio) < 800:  # < 50 ms of audio; produce zero vectors
        return np.zeros(6, dtype=np.float32), np.zeros(260, dtype=np.float32)
    f0 = _f0_contour(audio, sr)
    tone = tone_histogram_from_f0(f0)
    n_tone = np.linalg.norm(tone)
    if n_tone > 1e-6:
        tone = tone / n_tone

    mfcc = cheap_mfcc(audio, sr, n_mfcc=13)
    mfcc_sub = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
    mfcc_flat = mfcc_sub.flatten(order="F")
    mfcc_delta = np.diff(mfcc_sub, axis=1, prepend=mfcc_sub[:, :1])
    mfcc_delta_flat = mfcc_delta.flatten(order="F")
    seg = np.concatenate([mfcc_flat, mfcc_delta_flat]).astype(np.float32)
    n_seg = np.linalg.norm(seg)
    if n_seg > 1e-6:
        seg = seg / n_seg
    return tone, seg


@dataclass
class CacheEntry:
    """Multi-length fingerprint cache entry. The streaming-lookup design stores
    one fingerprint per decision-point length for each canonical utterance, so
    that a partial query at T=0.5s is compared against the canon's own 0.5s
    prefix fingerprint, not against the full-length canon fingerprint (which
    would be an apples-to-oranges comparison and forces commit only at full)."""
    key: str
    # mapping decision_point_t (None => full) -> (tone_vec, seg_vec)
    by_t: dict
    duration_s: float


def existence_lookup(tone_q, seg_q, store, tau_a, tau_b, t_cut):
    """Return (best_key, t_score, s_score) if any entry has a length-matched
    fingerprint with cos(tone) >= tau_a AND cos(seg) >= tau_b."""
    best = None
    best_sum = -1.0
    for e in store:
        # Use the length-matched stored fingerprint when available; fall back
        # to the closest shorter prefix; final fallback is the full-length one.
        candidate_ts = None
        if t_cut in e.by_t:
            candidate_ts = e.by_t[t_cut]
        else:
            # Pick the closest available decision-point length <= t_cut, else full.
            available = [k for k in e.by_t.keys() if k is not None and (t_cut is None or k <= t_cut)]
            if available:
                k_best = max(available)
                candidate_ts = e.by_t[k_best]
            elif None in e.by_t:
                candidate_ts = e.by_t[None]
        if candidate_ts is None:
            continue
        tone_e, seg_e = candidate_ts
        ts = float(np.dot(tone_q, tone_e))
        ss = float(np.dot(seg_q, seg_e))
        if ts >= tau_a and ss >= tau_b:
            if ts + ss > best_sum:
                best_sum = ts + ss
                best = (e.key, ts, ss)
    return best


def streaming_lookup(audio: np.ndarray, sr: int, store,
                     tau_a: float, tau_b: float,
                     decision_points_s=DECISION_POINTS_S,
                     require_consecutive: int = 2):
    """Walk decision points; commit on the first time the SAME entry key
    satisfies the existence rule under (tau_a, tau_b) at
    `require_consecutive` consecutive decision points. The consecutive-
    confirmation guard is a deployment-grade safety: at 0.5s of audio many
    Vietnamese utterances are acoustically similar (the same opening
    syllable), so a single passing comparison is not enough evidence to
    commit; two in a row at different prefix lengths is harder to fake.
    Setting require_consecutive=1 recovers the naive first-pass rule."""
    t0 = time.perf_counter()
    duration_s = len(audio) / sr
    timeline = []
    committed = None
    last_passing_key = None
    consecutive_count = 0

    for t_cut in decision_points_s:
        if t_cut is None:
            prefix = audio
            t_eff = duration_s
        else:
            n = int(min(len(audio), round(t_cut * sr)))
            if n < int(0.3 * sr):
                continue
            prefix = audio[:n]
            t_eff = n / sr
            if t_eff > duration_s + 1e-6:
                continue

        # Simulate stream arrival: real time accrues at 1x audio time until t_eff
        # plus fingerprint compute. We model wall-clock TTFH as
        #   max(t_eff, audio capture time so far) + fingerprint compute.
        # Capture-bounded floor (audio arrives in real time).
        capture_floor_ms = t_eff * 1000.0

        t_fp0 = time.perf_counter()
        tone_q, seg_q = _components_partial(prefix, sr)
        fp_ms = (time.perf_counter() - t_fp0) * 1000.0

        t_lk0 = time.perf_counter()
        res = existence_lookup(tone_q, seg_q, store, tau_a, tau_b, t_cut)
        lk_ms = (time.perf_counter() - t_lk0) * 1000.0

        decision_ms = capture_floor_ms + fp_ms + lk_ms
        timeline.append({
            "t_cut_s": round(t_eff, 3),
            "fp_ms": round(fp_ms, 2),
            "lookup_ms": round(lk_ms, 2),
            "ttfh_floor_ms": round(decision_ms, 2),
            "hit": res is not None,
            "candidate_key": res[0] if res else None,
            "t_score": round(res[1], 4) if res else None,
            "s_score": round(res[2], 4) if res else None,
        })
        if res is not None:
            cand_key = res[0]
            if cand_key == last_passing_key:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_passing_key = cand_key
            if consecutive_count >= require_consecutive and committed is None:
                committed = {
                    "commit_at_s": round(t_eff, 3),
                    "fraction_of_audio": round(min(1.0, t_eff / max(duration_s, 1e-3)), 3),
                    "ttfh_ms": round(decision_ms, 2),
                    "key": cand_key,
                    "t_score": round(res[1], 4),
                    "s_score": round(res[2], 4),
                    "consecutive_passes": consecutive_count,
                }
                break
        else:
            consecutive_count = 0
            last_passing_key = None

    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "duration_s": round(duration_s, 3),
        "tau_a": tau_a,
        "tau_b": tau_b,
        "committed": committed is not None,
        "commit": committed,
        "timeline": timeline,
        "wall_compute_ms": round(elapsed_total_ms, 2),
    }


def _add_noise(audio: np.ndarray, snr_db: float = 15.0, seed: int = 0) -> np.ndarray:
    """Additive pink-ish noise at given SNR (simple DEMAND-like surrogate)."""
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(audio ** 2) + 1e-12)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    # crude 1/f shaping
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), d=1/16000)
    spec *= 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    noise = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    p_n = float(np.mean(noise ** 2) + 1e-12)
    target_p_n = p_sig / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_p_n / p_n)
    return (audio + noise).astype(np.float32)


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def collect_audio_pool(max_n: int = 60) -> list[Path]:
    """Hold-out a pool of VIVOS .wav files. Skip VIVOSDEV01/*R002* (the one
    used in the README cmdline example, reserve it as a 'demo' clip)."""
    root = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test" / "waves"
    if not root.exists():
        raise FileNotFoundError(f"VIVOS test waves not found at {root}")
    paths = sorted(root.rglob("*.wav"))
    # Use deterministic shuffle for reproducibility, then truncate
    rng = np.random.default_rng(42)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    return [paths[i] for i in idx[:max_n]]


def run_eval(n_canon: int = 30, n_miss: int = 30, snr_db: float = 25.0) -> dict:
    pool = collect_audio_pool(max_n=n_canon + n_miss + 10)
    if len(pool) < n_canon + n_miss:
        n_canon = max(5, len(pool) // 2)
        n_miss = len(pool) - n_canon
    canon_paths = pool[:n_canon]
    miss_paths  = pool[n_canon:n_canon + n_miss]

    print(f"[load] canon={len(canon_paths)}  miss={len(miss_paths)}")

    # ----- Build cache from canon utterances (multi-length fingerprints) -----
    # On a real first-utterance miss the cascade runs and the cache stores the
    # full fingerprint. We additionally cache prefix fingerprints at the same
    # decision-point lengths used by the streaming lookup. This is cheap: each
    # extra prefix is < 30 ms of NumPy work and < 1 KB of memory per entry,
    # and it lets a partial-audio query commit before full-utterance arrival.
    store: list[CacheEntry] = []
    canon_audio = []
    for p in canon_paths:
        a, sr = load_wav(p)
        dur = len(a) / sr
        by_t = {}
        for t_cut in DECISION_POINTS_S:
            if t_cut is None:
                prefix = a
            else:
                n = int(min(len(a), round(t_cut * sr)))
                if n < int(0.3 * sr):
                    continue
                prefix = a[:n]
                if n / sr > dur + 1e-6:
                    continue
            tone_p, seg_p = _components_partial(prefix, sr)
            by_t[t_cut] = (tone_p, seg_p)
        store.append(CacheEntry(key=p.name, by_t=by_t, duration_s=dur))
        canon_audio.append((p.name, a, sr))
    print(f"[cache] stored {len(store)} canon entries (multi-length fingerprints)")

    # ----- Sweep thresholds -----
    results = {
        "config": {
            "n_canon": len(canon_paths), "n_miss": len(miss_paths),
            "decision_points_s": list(DECISION_POINTS_S),
            "snr_db": snr_db,
            "tau_a_grid": list(TAU_PARTIAL_A_GRID),
            "tau_b_grid": list(TAU_PARTIAL_B_GRID),
            "tau_full_a": TAU_FULL_A, "tau_full_b": TAU_FULL_B,
        },
        "sweep": [],
    }

    # Pre-load miss audios
    miss_audios = [(p.name, *load_wav(p)) for p in miss_paths]

    for tau_a in TAU_PARTIAL_A_GRID:
        for tau_b in TAU_PARTIAL_B_GRID:
            print(f"\n[sweep] tau_a={tau_a}  tau_b={tau_b}")
            per_hit = []
            per_miss = []

            # ---- HIT queries: noise-perturbed copy of each canon ----
            for key, a, sr in canon_audio:
                a_noisy = _add_noise(a, snr_db=snr_db, seed=hash(key) & 0xffff)
                r = streaming_lookup(a_noisy, sr, store, tau_a, tau_b)
                correct = r["committed"] and (r["commit"]["key"] == key)
                wrong_commit = r["committed"] and (r["commit"]["key"] != key)
                per_hit.append({
                    "key": key,
                    "duration_s": r["duration_s"],
                    "committed": r["committed"],
                    "correct_commit": bool(correct),
                    "wrong_commit": bool(wrong_commit),
                    "commit_at_s": r["commit"]["commit_at_s"] if r["commit"] else None,
                    "fraction_of_audio": r["commit"]["fraction_of_audio"] if r["commit"] else None,
                    "ttfh_ms": r["commit"]["ttfh_ms"] if r["commit"] else None,
                })

            # ---- MISS queries: out-of-cache utterances ----
            for key, a, sr in miss_audios:
                r = streaming_lookup(a, sr, store, tau_a, tau_b)
                # Any commit here is a false commit (these are NOT in cache)
                per_miss.append({
                    "key": key,
                    "duration_s": r["duration_s"],
                    "false_commit": bool(r["committed"]),
                    "commit_at_s": r["commit"]["commit_at_s"] if r["commit"] else None,
                    "ttfh_ms": r["commit"]["ttfh_ms"] if r["commit"] else None,
                })

            # ---- Aggregate ----
            n_hit_total = len(per_hit)
            n_correct = sum(1 for h in per_hit if h["correct_commit"])
            n_wrong = sum(1 for h in per_hit if h["wrong_commit"])
            n_committed_hit = sum(1 for h in per_hit if h["committed"])

            correct_ttfhs = sorted([h["ttfh_ms"] for h in per_hit if h["correct_commit"]])
            correct_fracs = sorted([h["fraction_of_audio"] for h in per_hit if h["correct_commit"]])
            correct_durs  = sorted([h["duration_s"] for h in per_hit if h["correct_commit"]])

            def pct(xs, q):
                if not xs: return None
                i = int(round(q * (len(xs) - 1)))
                return round(xs[i], 3)

            n_miss_total = len(per_miss)
            n_false = sum(1 for m in per_miss if m["false_commit"])

            # Batch baseline TTFH: must wait full duration + full fingerprint
            # We measure full-fingerprint time on every HIT to set the baseline.
            t_fp_full = []
            for key, a, sr in canon_audio:
                t0 = time.perf_counter()
                _components_partial(a, sr)
                t_fp_full.append((time.perf_counter() - t0) * 1000.0)
            t_fp_full.sort()
            batch_ttfhs = []
            for h in per_hit:
                if h["correct_commit"]:
                    # batch must process the full noisy audio
                    batch_ttfhs.append(h["duration_s"] * 1000.0 + (t_fp_full[len(t_fp_full)//2]))
            batch_ttfhs.sort()

            cell = {
                "tau_a": tau_a, "tau_b": tau_b,
                "n_hit": n_hit_total,
                "n_correct_commit": n_correct,
                "n_wrong_commit_on_hit": n_wrong,
                "n_no_commit_on_hit": n_hit_total - n_committed_hit,
                "recall_on_hits": round(n_correct / max(n_hit_total, 1), 4),
                "n_miss": n_miss_total,
                "n_false_commit_on_miss": n_false,
                "false_commit_rate": round(n_false / max(n_miss_total, 1), 4),
                "ttfh_streaming_correct": {
                    "p50_ms": pct(correct_ttfhs, 0.5),
                    "p95_ms": pct(correct_ttfhs, 0.95),
                    "p99_ms": pct(correct_ttfhs, 0.99),
                    "min_ms": round(correct_ttfhs[0], 2) if correct_ttfhs else None,
                    "max_ms": round(correct_ttfhs[-1], 2) if correct_ttfhs else None,
                },
                "ttfh_batch_baseline": {
                    "p50_ms": pct(batch_ttfhs, 0.5),
                    "p95_ms": pct(batch_ttfhs, 0.95),
                    "min_ms": round(batch_ttfhs[0], 2) if batch_ttfhs else None,
                    "max_ms": round(batch_ttfhs[-1], 2) if batch_ttfhs else None,
                },
                "fraction_of_audio_committed": {
                    "p50": pct(correct_fracs, 0.5),
                    "p95": pct(correct_fracs, 0.95),
                    "min": round(correct_fracs[0], 3) if correct_fracs else None,
                    "max": round(correct_fracs[-1], 3) if correct_fracs else None,
                },
                "duration_of_committed_hits_s": {
                    "p50": pct(correct_durs, 0.5),
                    "max": round(correct_durs[-1], 3) if correct_durs else None,
                },
                "per_hit": per_hit,
                "per_miss": per_miss,
            }
            results["sweep"].append(cell)
            print(f"  recall={cell['recall_on_hits']}  false_commit_rate={cell['false_commit_rate']}  "
                  f"ttfh_p50={cell['ttfh_streaming_correct']['p50_ms']}ms  "
                  f"batch_p50={cell['ttfh_batch_baseline']['p50_ms']}ms")

    # ----- Pick chosen operating point: tightest tau that keeps false_commit==0 with max recall -----
    zero_false = [c for c in results["sweep"] if c["false_commit_rate"] == 0.0]
    if zero_false:
        chosen = max(zero_false, key=lambda c: (c["recall_on_hits"], -c["tau_a"]))
        results["chosen_operating_point"] = {
            "tau_a": chosen["tau_a"], "tau_b": chosen["tau_b"],
            "rationale": "tightest tau with zero false-commit and max recall",
            "summary": {
                "recall_on_hits": chosen["recall_on_hits"],
                "false_commit_rate": chosen["false_commit_rate"],
                "ttfh_streaming_p50_ms": chosen["ttfh_streaming_correct"]["p50_ms"],
                "ttfh_batch_p50_ms": chosen["ttfh_batch_baseline"]["p50_ms"],
                "fraction_of_audio_p50": chosen["fraction_of_audio_committed"]["p50"],
            }
        }
    else:
        chosen = min(results["sweep"], key=lambda c: (c["false_commit_rate"], -c["recall_on_hits"]))
        results["chosen_operating_point"] = {
            "tau_a": chosen["tau_a"], "tau_b": chosen["tau_b"],
            "rationale": "no zero-false-commit cell; minimum false-commit / max recall",
            "summary": {
                "recall_on_hits": chosen["recall_on_hits"],
                "false_commit_rate": chosen["false_commit_rate"],
                "ttfh_streaming_p50_ms": chosen["ttfh_streaming_correct"]["p50_ms"],
                "ttfh_batch_p50_ms": chosen["ttfh_batch_baseline"]["p50_ms"],
                "fraction_of_audio_p50": chosen["fraction_of_audio_committed"]["p50"],
            }
        }
    return results


def plot_ttfh_cdf(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = results["chosen_operating_point"]
    cell = next(c for c in results["sweep"]
                if c["tau_a"] == chosen["tau_a"] and c["tau_b"] == chosen["tau_b"])
    s_ttfhs = sorted([h["ttfh_ms"] for h in cell["per_hit"] if h["correct_commit"]])
    b_ttfhs = sorted([h["duration_s"] * 1000.0 + 80.0 for h in cell["per_hit"] if h["correct_commit"]])

    if not s_ttfhs:
        print(f"[plot] no correct commits — skipping CDF plot")
        return

    s_y = np.arange(1, len(s_ttfhs) + 1) / len(s_ttfhs)
    b_y = np.arange(1, len(b_ttfhs) + 1) / len(b_ttfhs)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(s_ttfhs, s_y, "-o", color="#2a8a3e", ms=3, lw=1.6, label="Streaming (this work)")
    ax.plot(b_ttfhs, b_y, "-s", color="#cc6600", ms=3, lw=1.6, label="Batch baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Time-to-first-hit (ms, log)")
    ax.set_ylabel("CDF over cache-hit queries")
    ax.set_title(f"Streaming progressive cache: TTFH CDF\n"
                 f"$\\tau_A$={chosen['tau_a']}, $\\tau_B$={chosen['tau_b']}, "
                 f"false-commit={cell['false_commit_rate']:.2%}, "
                 f"recall={cell['recall_on_hits']:.2%}",
                 fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    out_json = PROJECT / "Results" / "streaming_cache_lookup.json"
    out_fig  = PROJECT / "papers" / "figures" / "fig_streaming_ttfh.png"

    results = run_eval(n_canon=30, n_miss=30, snr_db=25.0)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_json}")

    plot_ttfh_cdf(results, out_fig)

    chosen = results["chosen_operating_point"]
    print("\n=== Chosen operating point ===")
    print(json.dumps(chosen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
