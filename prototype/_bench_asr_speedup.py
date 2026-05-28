"""Micro-benchmark for ASR-only latency on pipeline_fast.

Usage:
    python prototype/_bench_asr_speedup.py --label baseline
    python prototype/_bench_asr_speedup.py --label final --merge
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent

AUDIOS = [
    "VIVOSDEV01/VIVOSDEV01_R099.wav",
    "VIVOSDEV02/VIVOSDEV02_R076.wav",
    "VIVOSDEV07/VIVOSDEV07_084.wav",
    "VIVOSDEV08/VIVOSDEV08_005.wav",
    "VIVOSDEV19/VIVOSDEV19_257.wav",
]
WAVES_ROOT = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test" / "waves"


def run_bench(trials: int = 3, threads: int | None = None):
    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
    cfg = FastPipelineConfig()
    if threads is not None:
        cfg.ct2_intra_threads = threads
    pipe = FastCascadeS2ST(cfg)
    pipe._ensure_asr()
    # Warmup (one shot) so first-call JIT is excluded from the timing aggregate.
    warm_audio, warm_sr = sf.read(str(WAVES_ROOT / AUDIOS[0]), dtype="float32")
    if warm_audio.ndim > 1:
        warm_audio = warm_audio.mean(axis=1)
    _ = pipe.transcribe(warm_audio.astype(np.float32), warm_sr)

    per_audio = {}
    all_lat = []
    all_text = []
    for rel in AUDIOS:
        audio, sr = sf.read(str(WAVES_ROOT / rel), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        lats = []
        texts = []
        for _ in range(trials):
            t0 = time.perf_counter()
            txt = pipe.transcribe(audio, sr)
            dt = (time.perf_counter() - t0) * 1000
            lats.append(dt)
            texts.append(txt)
        per_audio[rel] = {"lat_ms": lats, "text": texts[0]}
        all_lat.extend(lats)
        all_text.append(texts[0])
    p50 = float(np.percentile(all_lat, 50))
    p95 = float(np.percentile(all_lat, 95))
    mean = float(np.mean(all_lat))
    return {
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "mean_ms": round(mean, 1),
        "n_samples": len(all_lat),
        "per_audio": per_audio,
        "config": {
            "ct2_intra_threads": cfg.ct2_intra_threads,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=str(PROJECT / "Results" / "asr_speedup_bench.json"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    res = run_bench(args.trials, threads=args.threads)
    print(f"[bench {args.label}] p50={res['p50_ms']} ms  mean={res['mean_ms']} ms  p95={res['p95_ms']} ms")

    out = Path(args.out)
    data = {}
    if args.merge and out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
    data[args.label] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
