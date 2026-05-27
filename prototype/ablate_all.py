"""
Reproducible ablation bench — toggles each optimization individually and
measures end-to-end latency + WER + TER on a fixed VIVOS sample set.

Run:
    python prototype/ablate_all.py --n 5 --target en --out Results/ablation_n5.json

What it sweeps (each entry independently disable-able via env-vars):
  - FP32 HF cascade baseline                  (everything off)
  - INT8 cascade                              (INT8 on, no caches/VAD)
  - INT8 + tone-VAD                           (+ noise gating)
  - INT8 + T1 audio cache                     (+ tonal-perceptual fingerprint)
  - INT8 + T2 translation memory              (+ text-level fuzzy cache)
  - INT8 + T1 + T2                            (both caches)
  - INT8 + T1 + T2 + pre-warm                 (+ industrial phrase corpus)
  - FULL (all features)                       (everything on)

Output schema:
    {
      "config": {...},
      "configs_tested": [...],
      "per_config_results": [{config_id, samples, aggregate}, ...],
      "headline_table": [...]
    }

Why this matters: each config = one ablation row in the paper §4 table.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import soundfile as sf                                       # noqa: E402
import exp00_ter_metric as ter_mod                            # noqa: E402

PROJECT = HERE.parent
VIVOS_TEST = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"


def _load_vivos_test() -> dict[str, tuple[Path, str]]:
    prompts = {}
    pf = VIVOS_TEST / "prompts.txt"
    for line in pf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        spk = utt_id.split("_")[0]
        wav = VIVOS_TEST / "waves" / spk / f"{utt_id}.wav"
        if wav.exists():
            prompts[utt_id] = (wav, text.strip())
    return prompts


def _pct(xs, p):
    if not xs: return float("nan")
    s = sorted(xs)
    return s[int(round(p/100 * (len(s) - 1)))]


def _stats(xs):
    if not xs: return {}
    return {"mean": round(statistics.mean(xs), 1),
            "p50": round(_pct(xs, 50), 1),
            "p95": round(_pct(xs, 95), 1)}


CONFIGS = [
    # (config_id, description, env_overrides, pipeline_class)
    ("fp32_baseline",     "FP32 HF cascade (no optimizations)",
        {"VOICEAI_CACHE": "0", "VOICEAI_TM": "0", "VOICEAI_TONE_VAD": "0"}, "baseline"),
    ("int8_cascade",      "INT8 cascade alone",
        {"VOICEAI_CACHE": "0", "VOICEAI_TM": "0", "VOICEAI_TONE_VAD": "0"}, "fast"),
    ("int8_tone_vad",     "INT8 + tone-VAD",
        {"VOICEAI_CACHE": "0", "VOICEAI_TM": "0", "VOICEAI_TONE_VAD": "1"}, "fast"),
    ("int8_t1_only",      "INT8 + T1 audio cache",
        {"VOICEAI_CACHE": "1", "VOICEAI_TM": "0", "VOICEAI_TONE_VAD": "0"}, "fast"),
    ("int8_t2_only",      "INT8 + T2 text cache",
        {"VOICEAI_CACHE": "0", "VOICEAI_TM": "1", "VOICEAI_TONE_VAD": "0"}, "fast"),
    ("int8_t1_t2",        "INT8 + T1 + T2",
        {"VOICEAI_CACHE": "1", "VOICEAI_TM": "1", "VOICEAI_TONE_VAD": "0"}, "fast"),
    ("full",              "FULL stack (all on)",
        {"VOICEAI_CACHE": "1", "VOICEAI_TM": "1", "VOICEAI_TONE_VAD": "1"}, "fast"),
]


def _build_pipe(config_kind: str, env_overrides: dict, prewarm: bool, target: str):
    for k, v in env_overrides.items():
        os.environ[k] = v
    if config_kind == "baseline":
        from pipeline import CascadeS2ST, PipelineConfig
        pipe = CascadeS2ST(PipelineConfig())
        pipe._ensure_asr()
        pipe._ensure_mt()
        pipe._ensure_tts(target)
        return pipe, "baseline"
    else:
        # Re-import fresh so env vars take effect
        if "pipeline_fast" in sys.modules:
            del sys.modules["pipeline_fast"]
        from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
        pipe = FastCascadeS2ST(FastPipelineConfig())
        pipe._ensure_asr()
        pipe._ensure_mt()
        pipe._ensure_tts(target)
        if prewarm and pipe._tm is not None:
            try:
                from preload_cache import warm_translation_memory
                warm_translation_memory(pipe, target, include_audio=False, verbose=False)
            except Exception as e:
                print(f"  [warn] prewarm failed: {e}")
        return pipe, "fast"


def _bench_one(pipe, kind: str, samples: list, target: str) -> dict:
    rows = []
    for utt_id, wav, ref in samples:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        t0 = time.perf_counter()
        if kind == "baseline":
            res = pipe.run(audio, sr, target)
            vi = res["vi_text"]; lat = res["latency"]; cache_hit = False
        else:
            res = pipe.run(audio, sr, target)
            vi = res["vi_text"]; lat = res["latency"]
            cache_hit = res["cache"]["hit"]
        wall_ms = (time.perf_counter() - t0) * 1000

        st = ter_mod.ScoreState()
        st.add(ter_mod.normalize_vi(ref), ter_mod.normalize_vi(vi))
        s = st.summary()
        rows.append({
            "utt_id":           utt_id,
            "duration_s":       round(dur, 3),
            "wall_ms":          round(wall_ms, 1),
            "cache_hit":        cache_hit,
            "wer":              round(s["wer"], 4),
            "ter":              round(s["ter"], 4),
            "vi_hyp":           vi,
        })

    walls = [r["wall_ms"] for r in rows]
    return {
        "samples": rows,
        "aggregate": {
            "n":         len(rows),
            "latency":   _stats(walls),
            "wer_mean":  round(statistics.mean([r["wer"] for r in rows]), 4),
            "ter_mean":  round(statistics.mean([r["ter"] for r in rows]), 4),
            "cache_hit_rate": sum(1 for r in rows if r["cache_hit"]) / max(len(rows), 1),
        }
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=5)
    ap.add_argument("--target",  default="en")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     default=None)
    ap.add_argument("--skip-fp32", action="store_true", help="Skip the slow FP32 baseline (saves ~5 min)")
    args = ap.parse_args()

    all_samples = _load_vivos_test()
    rng = random.Random(args.seed)
    chosen_ids = rng.sample(list(all_samples.keys()), min(args.n, len(all_samples)))
    samples = [(i, *all_samples[i]) for i in chosen_ids]
    print(f"[plan] {len(samples)} VIVOS samples × {len(CONFIGS)} configs, target={args.target}")

    per_config_results = []
    for cfg_id, desc, env_overrides, kind in CONFIGS:
        if args.skip_fp32 and cfg_id == "fp32_baseline":
            continue
        prewarm = (cfg_id == "full")
        print(f"\n===== config: {cfg_id}  ({desc}) =====")
        print(f"      env: {env_overrides}  prewarm={prewarm}")
        t_setup = time.perf_counter()
        pipe, kind_resolved = _build_pipe(kind, env_overrides, prewarm, args.target)
        print(f"      pipeline ready in {time.perf_counter()-t_setup:.1f}s")
        result = _bench_one(pipe, kind_resolved, samples, args.target)
        result["config_id"] = cfg_id
        result["description"] = desc
        result["env"] = env_overrides
        per_config_results.append(result)
        agg = result["aggregate"]
        print(f"      latency: mean={agg['latency'].get('mean',0):.0f}ms  "
              f"p50={agg['latency'].get('p50',0):.0f}ms  p95={agg['latency'].get('p95',0):.0f}ms  "
              f"wer={agg['wer_mean']:.3f}  ter={agg['ter_mean']:.3f}  "
              f"cache_hit_rate={agg['cache_hit_rate']:.2f}")

    # Headline table
    headline = []
    baseline_mean = None
    for r in per_config_results:
        m = r["aggregate"]["latency"].get("mean", 0)
        if baseline_mean is None:
            baseline_mean = m
        speedup = (baseline_mean or m) / max(m, 1)
        headline.append({
            "config_id":      r["config_id"],
            "description":    r["description"],
            "latency_mean_ms": m,
            "latency_p50_ms": r["aggregate"]["latency"].get("p50", 0),
            "wer_mean":       r["aggregate"]["wer_mean"],
            "ter_mean":       r["aggregate"]["ter_mean"],
            "speedup":        round(speedup, 2),
            "cache_hit_rate": r["aggregate"]["cache_hit_rate"],
        })

    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY (n={args.n}, target={args.target})")
    print("=" * 90)
    print(f"{'config':<24}{'latency mean':>14}{'p50':>10}{'WER':>8}{'TER':>8}{'speedup':>10}{'hit%':>7}")
    print("-" * 90)
    for h in headline:
        print(f"  {h['config_id']:<22} {h['latency_mean_ms']:>10.0f} ms "
              f"{h['latency_p50_ms']:>8.0f} {h['wer_mean']:>7.3f} {h['ter_mean']:>7.3f} "
              f"{h['speedup']:>8.2f}x {h['cache_hit_rate']*100:>5.0f}%")
    print("=" * 90)

    out_path = Path(args.out or f"Results/ablation_{args.target}_n{args.n}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"n": args.n, "target": args.target, "seed": args.seed},
        "configs_tested": [c[0] for c in CONFIGS if not (args.skip_fp32 and c[0] == "fp32_baseline")],
        "headline_table": headline,
        "per_config_results": per_config_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
