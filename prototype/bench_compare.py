"""
Side-by-side latency bench: FP32 HF baseline (pipeline.py) vs INT8 CT2 (pipeline_fast.py).

Both pipelines are FULLY WARMED before timing — model load + first-call JIT
amortized OUTSIDE the measurement window so we measure steady-state inference.

Run:
    python prototype/bench_compare.py --n 10 --target en --out Results/bench_compare_en_n10.json

Output: JSON + console table with per-stage p50/p95/p99 + WER + speedup factor.
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

# UTF-8 stdout for Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

from pipeline import CascadeS2ST, PipelineConfig                # noqa: E402
from pipeline_fast import FastCascadeS2ST, FastPipelineConfig   # noqa: E402
import exp00_ter_metric as ter_mod                               # noqa: E402
import soundfile as sf                                            # noqa: E402

PROJECT = HERE.parent
VIVOS_TEST = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"


def _load_vivos_test() -> dict[str, tuple[Path, str]]:
    prompts = {}
    pf = VIVOS_TEST / "prompts.txt"
    if not pf.exists():
        raise FileNotFoundError(f"VIVOS test prompts not found at {pf}")
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
    return {
        "mean": round(statistics.mean(xs), 1),
        "std":  round(statistics.stdev(xs), 1) if len(xs) > 1 else 0.0,
        "p50":  round(_pct(xs, 50), 1),
        "p95":  round(_pct(xs, 95), 1),
        "p99":  round(_pct(xs, 99), 1),
        "min":  round(min(xs), 1),
        "max":  round(max(xs), 1),
    }


def warm_baseline(pipe: CascadeS2ST, target: str, sample_wav: Path) -> None:
    print(f"[warmup] FP32 baseline... (will materialize all 3 stages on a throwaway utterance)")
    audio, sr = sf.read(str(sample_wav), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    t = time.perf_counter()
    pipe._ensure_asr()
    pipe._ensure_mt()
    pipe._ensure_tts(target)
    # One throwaway run to trigger any JIT
    pipe.transcribe(audio[:int(sr)], sr)
    print(f"[warmup] FP32 ready in {time.perf_counter()-t:.1f}s")


def warm_fast(pipe: FastCascadeS2ST, target: str, sample_wav: Path) -> None:
    print(f"[warmup] INT8 fast... (loading CT2 models + INT8-quantizing TTS Linear layers)")
    audio, sr = sf.read(str(sample_wav), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    t = time.perf_counter()
    pipe._ensure_asr()
    pipe._ensure_mt()
    pipe._ensure_tts(target)
    # Throwaway inference
    pipe.transcribe(audio[:int(sr)], sr)
    print(f"[warmup] INT8 ready in {time.perf_counter()-t:.1f}s")


def run_one(pipe, audio, sr, target, st_class):
    t0 = time.perf_counter()
    if isinstance(pipe, FastCascadeS2ST):
        # Inside FastCascadeS2ST we have an optional background TTS preload — turn it off
        # for benchmarking so all CPU is used by foreground work.
        pipe.cfg.load_parallel = False
    if isinstance(pipe, CascadeS2ST):
        vi = pipe.transcribe(audio, sr)
    else:
        vi = pipe.transcribe(audio, sr)
    t1 = time.perf_counter()
    tgt = pipe.translate(vi, target, source="vi")
    t2 = time.perf_counter()
    _, _ = pipe.synthesize(tgt, target)
    t3 = time.perf_counter()
    return vi, tgt, {"asr": (t1-t0)*1000, "mt": (t2-t1)*1000, "tts": (t3-t2)*1000, "total": (t3-t0)*1000}


def bench(pipe, name: str, samples: list, target: str) -> dict:
    """Run timed loop on already-warmed pipeline."""
    print(f"\n===== {name} bench (n={len(samples)}, target={target}) =====")
    rows, lats_total, wers, ters, rtfs, durs = [], [], [], [], [], []
    for i, (utt_id, wav, ref) in enumerate(samples, start=1):
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        dur = len(audio)/sr
        vi, tgt, lat = run_one(pipe, audio, sr, target, ter_mod)
        st = ter_mod.ScoreState()
        st.add(ter_mod.normalize_vi(ref), ter_mod.normalize_vi(vi))
        s = st.summary()
        rows.append({"utt_id": utt_id, "duration_s": round(dur,3), "vi_ref": ref,
                     "vi_hyp": vi, "tgt": tgt, "wer": s["wer"], "ter": s["ter"],
                     "latency_ms": {k: round(v,1) for k,v in lat.items()}})
        lats_total.append(lat["total"])
        wers.append(s["wer"]); ters.append(s["ter"])
        durs.append(dur); rtfs.append(lat["total"]/(dur*1000))
        print(f"  [{i:>2}/{len(samples)}] {utt_id}  total={lat['total']:>7.0f}ms  "
              f"asr={lat['asr']:>6.0f}  mt={lat['mt']:>6.0f}  tts={lat['tts']:>6.0f}  "
              f"wer={s['wer']:.2f}  ter={s['ter']:.2f}")

    agg = {
        "n":                  len(rows),
        "latency_ms_asr":     _stats([r["latency_ms"]["asr"]   for r in rows]),
        "latency_ms_mt":      _stats([r["latency_ms"]["mt"]    for r in rows]),
        "latency_ms_tts":     _stats([r["latency_ms"]["tts"]   for r in rows]),
        "latency_ms_total":   _stats([r["latency_ms"]["total"] for r in rows]),
        "rtf":                _stats(rtfs),
        "wer_mean":           round(statistics.mean(wers), 4),
        "ter_mean":           round(statistics.mean(ters), 4),
        "audio_total_seconds":round(sum(durs), 1),
    }
    return {"samples": rows, "aggregate": agg}


def print_table(base_agg: dict, fast_agg: dict) -> None:
    print("\n" + "=" * 78)
    print(f"COMPARISON (n={base_agg['n']}, FP32 HF baseline vs INT8 CT2 fast)")
    print("=" * 78)
    print(f"{'metric':<22}{'FP32 baseline':>20}{'INT8 fast':>16}{'speedup':>12}{'Δ quality':>10}")
    print("-" * 78)
    for stat in ["p50", "p95", "p99", "mean"]:
        for name in ["asr", "mt", "tts", "total"]:
            b = base_agg[f"latency_ms_{name}"][stat]
            f = fast_agg[f"latency_ms_{name}"][stat]
            sp = b / max(f, 1e-9)
            print(f"  {name+' '+stat:<20}{b:>15.0f} ms{f:>13.0f} ms{sp:>10.2f}x")
        print()
    print(f"  {'RTF p50':<20}{base_agg['rtf']['p50']:>18.2f}{fast_agg['rtf']['p50']:>16.2f}"
          f"{base_agg['rtf']['p50']/max(fast_agg['rtf']['p50'],1e-9):>10.2f}x")
    print(f"  {'RTF p95':<20}{base_agg['rtf']['p95']:>18.2f}{fast_agg['rtf']['p95']:>16.2f}"
          f"{base_agg['rtf']['p95']/max(fast_agg['rtf']['p95'],1e-9):>10.2f}x")
    print()
    print(f"  {'WER mean':<20}{base_agg['wer_mean']:>20.4f}{fast_agg['wer_mean']:>16.4f}"
          f"{'':>12}{(fast_agg['wer_mean']-base_agg['wer_mean']):>+10.4f}")
    print(f"  {'TER mean':<20}{base_agg['ter_mean']:>20.4f}{fast_agg['ter_mean']:>16.4f}"
          f"{'':>12}{(fast_agg['ter_mean']-base_agg['ter_mean']):>+10.4f}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=10)
    ap.add_argument("--target",  default="en", choices=["en", "zh", "ko", "vi"])
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     default=None)
    ap.add_argument("--no-baseline", action="store_true", help="Skip FP32 baseline (just bench fast)")
    args = ap.parse_args()

    all_samples = _load_vivos_test()
    rng = random.Random(args.seed)
    ids = rng.sample(list(all_samples.keys()), min(args.n, len(all_samples)))
    samples = [(i, *all_samples[i]) for i in ids]
    warmup_sample = samples[0]

    # ---- Warm both pipelines first ----
    print("\n[stage] Loading + warming pipelines (NOT timed) ...")
    base_pipe = None
    if not args.no_baseline:
        base_pipe = CascadeS2ST(PipelineConfig())
        warm_baseline(base_pipe, args.target, warmup_sample[1])

    fast_pipe = FastCascadeS2ST(FastPipelineConfig())
    warm_fast(fast_pipe, args.target, warmup_sample[1])

    # ---- Run timed bench ----
    base_result = bench(base_pipe, "FP32 HF baseline", samples, args.target) if base_pipe else None
    fast_result = bench(fast_pipe, "INT8 CT2 fast",     samples, args.target)

    if base_result:
        print_table(base_result["aggregate"], fast_result["aggregate"])

    # ---- Save JSON ----
    out_path = Path(args.out or f"Results/bench_compare_{args.target}_n{args.n}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"n": args.n, "target": args.target, "seed": args.seed,
                   "baseline": asdict(PipelineConfig()) if base_pipe else None,
                   "fast":     asdict(FastPipelineConfig())},
        "baseline": base_result,
        "fast":     fast_result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
