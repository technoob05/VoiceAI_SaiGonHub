"""
Batch latency + WER/TER benchmark for the cascade pipeline.

Run from project root:
    python prototype/bench.py --n 20 --target en
    python prototype/bench.py --n 50 --target en --out Results/bench_phowhisper_fp32.json

What it does:
1. Picks N random VIVOS test utterances (seeded for reproducibility).
2. Runs each through CascadeS2ST (one stage at a time, measuring per-stage latency).
3. Computes ASR WER + Tone-Error-Rate (TER) using exp00's metric helpers.
4. Emits per-utterance JSON list + aggregate p50 / p95 / p99 / mean / std for ASR / MT / TTS / total.

This is the FP32 baseline. Re-run with quantized models from exp/expNN_*.py later
to compare.

Output JSON schema:
{
  "config":   {"model_asr": ..., "model_mt": ..., "tts_lang": "en", "n": 20, "seed": 42},
  "samples":  [{utt_id, vi_ref, vi_hyp, tgt_text, wer, ter,
                latency_ms: {asr, mt, tts, total}, audio_duration_s}, ...],
  "aggregate": {"latency_ms": {asr_p50, asr_p95, asr_p99, ...},
                "wer_mean": ..., "ter_mean": ..., ...},
}
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
from pipeline import CascadeS2ST, PipelineConfig           # noqa: E402
import exp00_ter_metric as ter_mod                          # noqa: E402

import soundfile as sf                                       # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

PROJECT = HERE.parent
VIVOS_TEST = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"


def _load_vivos_test() -> dict[str, tuple[Path, str]]:
    """Return {utt_id: (wav_path, ref_text)} from VIVOS test split."""
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


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int, default=20,                help="Number of samples (random)")
    ap.add_argument("--target", choices=["en", "zh", "ko", "vi"], default="en")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--out",    default=None,                         help="JSON output path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--asr-id", default="vinai/PhoWhisper-small")
    ap.add_argument("--mt-id",  default="facebook/nllb-200-distilled-600M")
    args = ap.parse_args()

    samples = _load_vivos_test()
    if not samples:
        print("[fatal] no VIVOS test samples found — did you extract data/processed/vivos/?")
        return 2
    print(f"[data] VIVOS test set: {len(samples)} utterances available")

    rng = random.Random(args.seed)
    chosen_ids = rng.sample(list(samples.keys()), min(args.n, len(samples)))
    print(f"[plan] benching n={len(chosen_ids)} samples, target={args.target}, seed={args.seed}")

    cfg = PipelineConfig(asr_id=args.asr_id, mt_id=args.mt_id, device=args.device)
    pipe = CascadeS2ST(cfg)
    print("[init] warming models (one-time)...")
    pipe._ensure_asr()
    pipe._ensure_mt()
    pipe._ensure_tts(args.target)
    print("[init] all stages warm")

    per_sample = []
    t_overall = time.perf_counter()

    for i, utt_id in enumerate(chosen_ids, start=1):
        wav_path, ref = samples[utt_id]
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr

        t0 = time.perf_counter()
        vi_hyp = pipe.transcribe(audio, sr)
        t1 = time.perf_counter()
        tgt_text = pipe.translate(vi_hyp, args.target, source="vi")
        t2 = time.perf_counter()
        _, _ = pipe.synthesize(tgt_text, args.target)
        t3 = time.perf_counter()

        # Metrics on ASR step (vi_hyp vs ref) — translation has no reference here.
        st = ter_mod.ScoreState()
        st.add(ter_mod.normalize_vi(ref), ter_mod.normalize_vi(vi_hyp))
        s = st.summary()

        entry = {
            "utt_id":            utt_id,
            "audio_duration_s":  round(dur, 3),
            "vi_ref":            ref,
            "vi_hyp":            vi_hyp,
            "tgt_text":          tgt_text,
            "wer":               round(s["wer"], 4),
            "ter":               round(s["ter"], 4),
            "ter_given_base_ok": round(s["ter_given_base_ok"], 4),
            "n_syllables":       s["ref_syllables"],
            "latency_ms": {
                "asr":   round((t1 - t0) * 1000, 1),
                "mt":    round((t2 - t1) * 1000, 1),
                "tts":   round((t3 - t2) * 1000, 1),
                "total": round((t3 - t0) * 1000, 1),
            },
        }
        per_sample.append(entry)
        print(f"  [{i:>3}/{len(chosen_ids)}] {utt_id}  wer={s['wer']:.2f}  ter={s['ter']:.2f}  "
              f"asr={entry['latency_ms']['asr']:>6.0f}ms  mt={entry['latency_ms']['mt']:>6.0f}ms  "
              f"tts={entry['latency_ms']['tts']:>6.0f}ms  total={entry['latency_ms']['total']:>6.0f}ms")

    wall = time.perf_counter() - t_overall

    # ---- aggregate ----
    asr   = [x["latency_ms"]["asr"]   for x in per_sample]
    mt    = [x["latency_ms"]["mt"]    for x in per_sample]
    tts   = [x["latency_ms"]["tts"]   for x in per_sample]
    tot   = [x["latency_ms"]["total"] for x in per_sample]
    wers  = [x["wer"] for x in per_sample]
    ters  = [x["ter"] for x in per_sample]
    durs  = [x["audio_duration_s"] for x in per_sample]
    rtfs  = [x["latency_ms"]["total"] / (x["audio_duration_s"] * 1000) for x in per_sample]

    def stats(xs: list[float]) -> dict:
        return {
            "mean": round(statistics.mean(xs), 2),
            "std":  round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
            "p50":  round(_percentile(xs, 50), 2),
            "p95":  round(_percentile(xs, 95), 2),
            "p99":  round(_percentile(xs, 99), 2),
            "min":  round(min(xs), 2),
            "max":  round(max(xs), 2),
        }

    agg = {
        "latency_ms_asr":   stats(asr),
        "latency_ms_mt":    stats(mt),
        "latency_ms_tts":   stats(tts),
        "latency_ms_total": stats(tot),
        "rtf":              stats(rtfs),
        "wer_mean":         round(statistics.mean(wers), 4),
        "ter_mean":         round(statistics.mean(ters), 4),
        "audio_total_seconds": round(sum(durs), 1),
        "wall_clock_seconds":  round(wall, 1),
    }

    print("\n" + "=" * 70)
    print(f"AGGREGATE (n={len(per_sample)}, target={args.target}, device={args.device})")
    print("=" * 70)
    print(f"  ASR   latency ms  p50={agg['latency_ms_asr']['p50']:>7.0f}  p95={agg['latency_ms_asr']['p95']:>7.0f}  p99={agg['latency_ms_asr']['p99']:>7.0f}  mean={agg['latency_ms_asr']['mean']:>7.0f}")
    print(f"  MT    latency ms  p50={agg['latency_ms_mt']['p50']:>7.0f}  p95={agg['latency_ms_mt']['p95']:>7.0f}  p99={agg['latency_ms_mt']['p99']:>7.0f}  mean={agg['latency_ms_mt']['mean']:>7.0f}")
    print(f"  TTS   latency ms  p50={agg['latency_ms_tts']['p50']:>7.0f}  p95={agg['latency_ms_tts']['p95']:>7.0f}  p99={agg['latency_ms_tts']['p99']:>7.0f}  mean={agg['latency_ms_tts']['mean']:>7.0f}")
    print(f"  TOTAL latency ms  p50={agg['latency_ms_total']['p50']:>7.0f}  p95={agg['latency_ms_total']['p95']:>7.0f}  p99={agg['latency_ms_total']['p99']:>7.0f}  mean={agg['latency_ms_total']['mean']:>7.0f}")
    print(f"  RTF               p50={agg['rtf']['p50']:>7.2f}  p95={agg['rtf']['p95']:>7.2f}  p99={agg['rtf']['p99']:>7.2f}  mean={agg['rtf']['mean']:>7.2f}")
    print(f"  WER   mean = {agg['wer_mean']:.4f}    TER mean = {agg['ter_mean']:.4f}")
    print(f"  Wall clock: {agg['wall_clock_seconds']:.1f}s for {agg['audio_total_seconds']:.1f}s of audio")

    result = {
        "config":   {**asdict(cfg), "target": args.target, "n": len(per_sample),
                     "seed": args.seed, "device": args.device},
        "samples":  per_sample,
        "aggregate": agg,
    }
    out = Path(args.out or f"Results/bench_phowhisper_fp32_{args.target}_n{len(per_sample)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
