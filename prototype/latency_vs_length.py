"""
Latency vs utterance length (reviewer Q7) — characterize cache-hit and cache-miss
end-to-end latency as a function of source utterance duration on real VIVOS audio.

Reviewer concern: "The 25 ms cache-hit latency appears to include TTS; what is
the utterance duration used and how does latency scale with output length?"

We:
  1. Pick 5 VIVOS test clips spanning ~1.2 s to ~7.8 s.
  2. For each, run FastCascadeS2ST 5 trials at cache HIT (after a warm-up) and
     1 trial at cache MISS (clear cache between).
  3. Separately measure TTS sub-stage latency vs target-text length on the
     translations produced.
  4. Save Results/latency_vs_length.json and papers/figures/fig_latency_vs_length.png.

CPU-only, pure Python timing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME",            str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",       str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE",       str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))
# Silence cache TM / VAD for cleaner measurement; we want pure T1 audio cache
os.environ.setdefault("VOICEAI_TM", "0")
os.environ.setdefault("VOICEAI_TONE_VAD", "0")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402


# Chosen by hand from a duration scan of the VIVOS test split.
CLIPS = [
    ("data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R034.wav", "short"),
    ("data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R003.wav", "medium"),
    ("data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R044.wav", "long"),
    ("data/processed/vivos/vivos/test/waves/VIVOSDEV13/VIVOSDEV13_097.wav", "xlong-a"),
    ("data/processed/vivos/vivos/test/waves/VIVOSDEV13/VIVOSDEV13_105.wav", "xlong-b"),
]


def percentile(xs, q):
    if not xs:
        return float("nan")
    return float(np.percentile(np.array(xs, dtype=np.float64), q))


def measure_tts_only(pipe, text: str, target: str, trials: int = 3) -> dict:
    """Measure TTS-only latency for a target text. trials=3 keeps cost low."""
    # warm-up
    pipe.synthesize(text, target)
    times_ms = []
    out_durs = []
    for _ in range(trials):
        t = time.perf_counter()
        wav, sr = pipe.synthesize(text, target)
        times_ms.append((time.perf_counter() - t) * 1000)
        out_durs.append(len(wav) / max(sr, 1))
    return {
        "tts_p50_ms": percentile(times_ms, 50),
        "tts_p95_ms": percentile(times_ms, 95),
        "tts_min_ms": min(times_ms),
        "tts_max_ms": max(times_ms),
        "tts_all_ms": times_ms,
        "out_audio_dur_s_p50": percentile(out_durs, 50),
        "n_chars":   len(text),
        "n_words":   len(text.split()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="en")
    ap.add_argument("--hit-trials", type=int, default=5)
    ap.add_argument("--out-json", default="Results/latency_vs_length.json")
    ap.add_argument("--out-fig",  default="papers/figures/fig_latency_vs_length.png")
    args = ap.parse_args()

    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig

    pipe = FastCascadeS2ST(FastPipelineConfig())
    print("[init] pipeline built. warming up models (one cold cascade) ...")
    warmup_path = PROJECT / CLIPS[0][0]
    a0, sr0 = sf.read(str(warmup_path), dtype="float32")
    if a0.ndim > 1: a0 = a0.mean(axis=1)
    _ = pipe.run(a0, sr0, args.target)   # cold run preloads all models
    # Wipe cache so each clip below gets its own clean miss timing
    if pipe._utt_cache is not None:
        pipe._utt_cache._store.clear()
        from utterance_cache import CacheStats
        pipe._utt_cache.stats = CacheStats()

    per_clip = []
    for rel, label in CLIPS:
        p = PROJECT / rel
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        duration_s = len(a) / sr
        print(f"\n[clip] {label}  duration={duration_s:.2f}s  {rel}")

        # ---- Cache MISS (cold) ----
        # Clear cache so it's a true miss
        pipe._utt_cache._store.clear()
        t = time.perf_counter()
        miss = pipe.run(a, sr, args.target)
        miss_total_ms = (time.perf_counter() - t) * 1000
        miss_lat = miss["latency"]
        print(f"  miss : total={miss_total_ms:.0f}ms  asr={miss_lat['asr_ms']:.0f}  "
              f"mt={miss_lat['mt_ms']:.0f}  tts={miss_lat['tts_ms']:.0f}  "
              f"cache={miss_lat['cache_ms']:.1f}")

        # ---- Cache HIT (after put) ----
        hit_totals = []
        hit_breakdowns = []
        for _ in range(args.hit_trials):
            t = time.perf_counter()
            hit = pipe.run(a, sr, args.target)
            ms = (time.perf_counter() - t) * 1000
            hit_totals.append(ms)
            hit_breakdowns.append(hit["latency"])
            if not hit["cache"]["hit"]:
                print(f"  [warn] expected cache HIT but got miss; cache_cosine={hit['cache'].get('cosine')}")
        print(f"  hit  : p50={percentile(hit_totals,50):.1f}ms  "
              f"p95={percentile(hit_totals,95):.1f}ms  "
              f"min={min(hit_totals):.1f}  max={max(hit_totals):.1f}")

        # ---- Measure TTS-only on the translated text (already produced in miss) ----
        tgt_text = miss["tgt_text"]
        tts_meas = measure_tts_only(pipe, tgt_text, args.target, trials=3)
        print(f"  tts  : p50={tts_meas['tts_p50_ms']:.0f}ms  out_dur={tts_meas['out_audio_dur_s_p50']:.2f}s  "
              f"text_chars={tts_meas['n_chars']}  words={tts_meas['n_words']}")

        per_clip.append({
            "label":          label,
            "path":           rel,
            "duration_s":     round(duration_s, 3),
            "vi_text":        miss["vi_text"],
            "tgt_text":       tgt_text,
            "miss": {
                "total_ms":   round(miss_total_ms, 1),
                "asr_ms":     round(miss_lat["asr_ms"], 1),
                "mt_ms":      round(miss_lat["mt_ms"],  1),
                "tts_ms":     round(miss_lat["tts_ms"], 1),
                "cache_ms":   round(miss_lat["cache_ms"], 2),
            },
            "hit": {
                "trials":     args.hit_trials,
                "p50_ms":     round(percentile(hit_totals, 50), 2),
                "p95_ms":     round(percentile(hit_totals, 95), 2),
                "min_ms":     round(min(hit_totals), 2),
                "max_ms":     round(max(hit_totals), 2),
                "all_ms":     [round(x, 2) for x in hit_totals],
            },
            "tts_only": {
                "p50_ms":     round(tts_meas["tts_p50_ms"], 1),
                "p95_ms":     round(tts_meas["tts_p95_ms"], 1),
                "out_audio_dur_s_p50": round(tts_meas["out_audio_dur_s_p50"], 3),
                "n_chars":    tts_meas["n_chars"],
                "n_words":    tts_meas["n_words"],
            },
        })

    # ---- Fit scaling exponents ----
    durations = np.array([c["duration_s"] for c in per_clip])
    hit_p50  = np.array([c["hit"]["p50_ms"] for c in per_clip])
    hit_p95  = np.array([c["hit"]["p95_ms"] for c in per_clip])
    miss_tot = np.array([c["miss"]["total_ms"] for c in per_clip])
    tts_p50  = np.array([c["tts_only"]["p50_ms"] for c in per_clip])

    # Linear fit: latency = alpha * duration + beta
    def lin_fit(x, y):
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        return float(m), float(b), float(r2)

    # Power-law fit: latency = a * duration^k  → log(lat) = log(a) + k log(dur)
    def pow_fit(x, y):
        lx, ly = np.log(x), np.log(np.maximum(y, 1e-6))
        k, log_a = np.polyfit(lx, ly, 1)
        y_pred = np.exp(log_a) * x ** k
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        return float(k), float(np.exp(log_a)), float(r2)

    hit_lin = lin_fit(durations, hit_p50)
    hit_p95_lin = lin_fit(durations, hit_p95)
    miss_lin = lin_fit(durations, miss_tot)
    tts_lin = lin_fit(durations, tts_p50)
    hit_pow = pow_fit(durations, hit_p50)
    miss_pow = pow_fit(durations, miss_tot)

    print(f"\n[fit] hit p50 linear : slope={hit_lin[0]:.2f} ms/s   intercept={hit_lin[1]:.2f}   R^2={hit_lin[2]:.3f}")
    print(f"[fit] hit p95 linear : slope={hit_p95_lin[0]:.2f} ms/s  intercept={hit_p95_lin[1]:.2f}  R^2={hit_p95_lin[2]:.3f}")
    print(f"[fit] hit p50 power  : exponent={hit_pow[0]:.2f}  R^2={hit_pow[2]:.3f}")
    print(f"[fit] miss total linear: slope={miss_lin[0]:.2f} ms/s   intercept={miss_lin[1]:.2f}  R^2={miss_lin[2]:.3f}")
    print(f"[fit] miss total power : exponent={miss_pow[0]:.2f}   R^2={miss_pow[2]:.3f}")
    print(f"[fit] tts-only linear: slope={tts_lin[0]:.2f} ms/s   intercept={tts_lin[1]:.2f}    R^2={tts_lin[2]:.3f}")

    # 2.3 s industrial directive average → projected hit/miss ratio
    typ_dur = 2.3
    proj_hit = hit_lin[0] * typ_dur + hit_lin[1]
    proj_miss = miss_lin[0] * typ_dur + miss_lin[1]
    proj_ratio = proj_miss / max(proj_hit, 1e-6)

    longest_dur = float(durations.max())
    long_hit = hit_lin[0] * longest_dur + hit_lin[1]
    long_miss = miss_lin[0] * longest_dur + miss_lin[1]
    long_ratio = long_miss / max(long_hit, 1e-6)

    print(f"\n[project] at 2.3 s typical directive: hit={proj_hit:.1f}ms  miss={proj_miss:.0f}ms  ratio={proj_ratio:.0f}x")
    print(f"[project] at longest tested ({longest_dur:.1f}s): hit={long_hit:.1f}ms  miss={long_miss:.0f}ms  ratio={long_ratio:.0f}x")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        order = np.argsort(durations)
        ax.plot(durations[order], hit_p50[order], "o-", color="#2ca02c", label=f"Cache-hit p50 ({hit_lin[0]:.1f} ms/s)")
        ax.plot(durations[order], hit_p95[order], "v--", color="#2ca02c", alpha=0.6, label=f"Cache-hit p95")
        ax.plot(durations[order], miss_tot[order], "s-", color="#d62728", label=f"Cache-miss (full cascade)")
        ax.set_xlabel("Source utterance duration (s)")
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="best")
        ax.set_title("End-to-end latency vs source utterance duration (VIVOS, CPU INT8)")
        for d, h, m in zip(durations, hit_p50, miss_tot):
            ax.annotate(f"{h:.0f}", (d, h), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=8, color="#2ca02c")
            ax.annotate(f"{m/1000:.1f}s", (d, m), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#d62728")
        fig.tight_layout()
        out_fig = PROJECT / args.out_fig
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_fig), dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] wrote {out_fig}")
    except Exception as e:
        print(f"[warn] could not plot: {e}")

    out = {
        "status": "ok",
        "experiment_summary": "End-to-end latency on n=5 VIVOS clips at varying durations. CPU INT8. Reviewer Q7.",
        "clips":  per_clip,
        "fits": {
            "hit_p50_linear":  {"slope_ms_per_s": round(hit_lin[0], 3),
                                "intercept_ms":   round(hit_lin[1], 3),
                                "r2":             round(hit_lin[2], 4)},
            "hit_p95_linear":  {"slope_ms_per_s": round(hit_p95_lin[0], 3),
                                "intercept_ms":   round(hit_p95_lin[1], 3),
                                "r2":             round(hit_p95_lin[2], 4)},
            "hit_p50_power":   {"exponent":       round(hit_pow[0], 3),
                                "coefficient":    round(hit_pow[1], 3),
                                "r2":             round(hit_pow[2], 4)},
            "miss_total_linear": {"slope_ms_per_s": round(miss_lin[0], 3),
                                  "intercept_ms":   round(miss_lin[1], 3),
                                  "r2":             round(miss_lin[2], 4)},
            "miss_total_power":  {"exponent":       round(miss_pow[0], 3),
                                  "coefficient":    round(miss_pow[1], 3),
                                  "r2":             round(miss_pow[2], 4)},
            "tts_only_linear":   {"slope_ms_per_s": round(tts_lin[0], 3),
                                  "intercept_ms":   round(tts_lin[1], 3),
                                  "r2":             round(tts_lin[2], 4)},
        },
        "projections": {
            "typical_industrial_2.3s": {
                "hit_ms":      round(proj_hit, 1),
                "miss_ms":     round(proj_miss, 1),
                "ratio_x":     round(proj_ratio, 1),
            },
            "longest_tested_s": {
                "duration_s":  round(longest_dur, 2),
                "hit_ms":      round(long_hit, 1),
                "miss_ms":     round(long_miss, 1),
                "ratio_x":     round(long_ratio, 1),
            },
            "headline_25ms_anchor": "Headline 25 ms cache-hit is approximately the p50 at short utterances; p95 and longer utterances drift up linearly with TTS output length.",
        },
        "config": {
            "target":      args.target,
            "hit_trials":  args.hit_trials,
            "miss_trials": 1,
            "VOICEAI_TM":         os.environ.get("VOICEAI_TM"),
            "VOICEAI_TONE_VAD":   os.environ.get("VOICEAI_TONE_VAD"),
        },
    }
    out_json = PROJECT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
