"""
Worker-scale simulation — cache hit rate as a function of fleet size.

Paper §4.4 anchor: in a deployment with N workers each speaking from a
limited set of common industrial phrases, what end-to-end median latency
should we expect?

Model: each worker speaks K_phrases unique phrases per shift, with each
phrase repeated R_repeats times. We simulate a stream of utterances and
measure:
  - T1 audio-cache hit rate (worker-internal repeats)
  - T2 text-cache hit rate (cross-worker phrase reuse)
  - Combined hit rate
  - Median effective latency

Data we use:
  - VIVOS train: 46 distinct speakers × variable phrase set per speaker
  - We treat each VIVOS speaker as one factory worker
  - For each worker, we sample K phrases from their utterance pool, then
    create a randomized event stream of R repetitions
  - Pass each event to the actual FastCascadeS2ST cache stack (audio = real
    VIVOS wav, NOT synthesized) and record hit/miss + latency

This is realistic: same speaker repeating same phrase = real audio cache
hit; different speaker same phrase = T2 text hit; new phrase = full
cascade.

Output: Results/worker_simulation.json with hit rate at each scale.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
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

import soundfile as sf                                         # noqa: E402


VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"


def _gather_vivos_speakers() -> dict[str, list[Path]]:
    """Return {speaker_id: [wav_paths]} from VIVOS train."""
    waves = VIVOS_TRAIN / "waves"
    out: dict[str, list[Path]] = {}
    if not waves.exists():
        return out
    for spk_dir in sorted(waves.iterdir()):
        if not spk_dir.is_dir(): continue
        wavs = sorted(spk_dir.glob("*.wav"))
        if wavs:
            out[spk_dir.name] = wavs
    return out


def simulate(pipe, n_workers: int, n_phrases_per_worker: int,
             n_repeats_per_phrase: int, seed: int = 42) -> dict:
    """Run one (n_workers, n_phrases, n_repeats) scenario through the cache."""
    speakers = _gather_vivos_speakers()
    rng = random.Random(seed)

    # Pick n_workers speakers
    worker_ids = list(speakers.keys())[:n_workers]
    if len(worker_ids) < n_workers:
        return {"error": f"only {len(worker_ids)} VIVOS speakers available; requested {n_workers}"}

    # Per worker: sample n_phrases distinct utterances
    worker_phrases = {}
    for wid in worker_ids:
        avail = speakers[wid]
        n = min(n_phrases_per_worker, len(avail))
        worker_phrases[wid] = rng.sample(avail, n)

    # Build event stream: each (worker, phrase) appears n_repeats times,
    # shuffled chronologically
    events = []
    for wid, phrases in worker_phrases.items():
        for p in phrases:
            for _ in range(n_repeats_per_phrase):
                events.append((wid, p))
    rng.shuffle(events)

    # Run through pipeline; we use pipe.run() not run_streaming to get a clean dict
    rows = []
    t_start = time.perf_counter()
    for wid, wav_path in events:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        t0 = time.perf_counter()
        res = pipe.run(audio, sr, "en")
        wall_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "worker":  wid,
            "phrase":  wav_path.stem,
            "wall_ms": round(wall_ms, 1),
            "t1_hit":  bool(res.get("cache", {}).get("hit", False)),
        })

    walls = [r["wall_ms"] for r in rows]
    walls.sort()
    n_t1_hits = sum(1 for r in rows if r["t1_hit"])
    # Compute t2 hits from pipeline's TM stats (only available end of run)
    tm_stats = pipe._tm.summary() if pipe._tm else None

    return {
        "config":        {"n_workers": n_workers, "n_phrases_per_worker": n_phrases_per_worker,
                          "n_repeats_per_phrase": n_repeats_per_phrase, "seed": seed,
                          "total_events": len(events)},
        "n_events":      len(events),
        "t1_hit_count":  n_t1_hits,
        "t1_hit_rate":   round(n_t1_hits / max(len(events), 1), 4),
        "tm_stats":      tm_stats,
        "latency_ms": {
            "mean":  round(sum(walls) / max(len(walls), 1), 1),
            "p50":   round(walls[len(walls)//2], 1) if walls else 0,
            "p95":   round(walls[int(len(walls)*0.95)], 1) if walls else 0,
            "p99":   round(walls[int(len(walls)*0.99)], 1) if walls else 0,
            "max":   round(max(walls), 1) if walls else 0,
        },
        "wall_clock_seconds": round(time.perf_counter() - t_start, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers",         nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--phrases-per-worker", type=int, default=2)
    ap.add_argument("--repeats",         type=int, default=3)
    ap.add_argument("--out",             default="Results/worker_simulation.json")
    args = ap.parse_args()

    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
    cfg = FastPipelineConfig(cache_enabled=True, tm_enabled=True, tone_vad_enabled=False)
    pipe = FastCascadeS2ST(cfg)
    print("[init] warming pipeline ...")
    pipe._ensure_asr()
    pipe._ensure_mt()
    pipe._ensure_tts("en")
    print("[init] ready")

    scenarios = []
    for w in args.workers:
        print(f"\n===== scenario: {w} workers × {args.phrases_per_worker} phrases × {args.repeats} repeats =====")
        # Reset caches between scenarios so each one is independent
        if pipe._utt_cache: pipe._utt_cache._store.clear()
        if pipe._tm:        pipe._tm._store.clear()
        res = simulate(pipe, w, args.phrases_per_worker, args.repeats)
        scenarios.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}"); continue
        lat = res["latency_ms"]
        print(f"  events={res['n_events']}  t1_hits={res['t1_hit_count']} "
              f"({res['t1_hit_rate']*100:.1f}%)  "
              f"latency p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms mean={lat['mean']:.0f}ms")
        if res.get("tm_stats"):
            tm = res["tm_stats"]
            print(f"  TM: lookups={tm['lookups']} hits={tm['hits']} ({tm['hit_rate']*100:.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "scenarios": scenarios,
        "global_config": {"phrases_per_worker": args.phrases_per_worker, "repeats": args.repeats},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
