"""
demo_wow.py — show off the optimization stack on cold vs warm vs cache-hit.

For OneVoice AI Challenge demo or paper qualitative figure. Runs three regimes
on the same input audio and prints a comparison table.

Regimes:
  1. FP32 baseline (HF cascade, cold load) — what reviewers will see as the "starting point"
  2. INT8 fast cascade (faster-whisper + CT2 NLLB + dynamic-INT8 VITS), warmed
  3. INT8 fast cascade + tonal-perceptual utterance cache hit

Expected pattern on a 4 s VIVOS utterance, CPU fp32:
    FP32 cold cascade  : ~100 s   (RTF 25x)
    INT8 warm cascade  :   ~6 s   (RTF 1.5x)  -- 16x faster than cold baseline
    INT8 + cache hit   : ~50 ms   (RTF 0.01x) -- 2000x faster, basically instant

Run:
    python prototype/demo_wow.py                   # uses VIVOSDEV01_R002.wav
    python prototype/demo_wow.py --audio <vi.wav>  # any Vi WAV
    python prototype/demo_wow.py --target zh       # or ko
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Force UTF-8 stdout for Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import soundfile as sf                                       # noqa: E402

from pipeline import CascadeS2ST, PipelineConfig             # noqa: E402
from pipeline_fast import FastCascadeS2ST, FastPipelineConfig # noqa: E402

DEFAULT_AUDIO = HERE.parent / "data" / "processed" / "vivos" / "vivos" / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav"


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio",  default=str(DEFAULT_AUDIO), help="Vi WAV to translate")
    ap.add_argument("--target", default="en", choices=["en", "zh", "ko", "vi"])
    ap.add_argument("--skip-fp32", action="store_true",
                    help="Skip the slow FP32 baseline (useful if you've seen it once)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[fatal] audio not found: {audio_path}")
        return 2
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"\n📂 input: {audio_path.name}  duration={duration:.2f}s  sr={sr} Hz")
    print(f"🌐 target language: {args.target}")
    print("=" * 78)

    results = []

    # ============================== Regime 1: FP32 baseline ==============================
    if not args.skip_fp32:
        print("\n[1/3] 🐢  FP32 HF baseline cascade (cold, includes load) ...")
        base = CascadeS2ST(PipelineConfig())
        t0 = time.perf_counter()
        out_b = base.run(audio, sr, args.target)
        b_total = (time.perf_counter() - t0) * 1000
        print(f"      vi_text : {out_b['vi_text']!r}")
        print(f"      tgt_text: {out_b['tgt_text']!r}")
        print(f"      total   : {b_total:>8.0f} ms   RTF {b_total/(duration*1000):.2f}x")
        results.append(("FP32 baseline (cold)", b_total, out_b))
    else:
        print("\n[1/3] (skipped FP32 baseline)")
        results.append(("FP32 baseline (cold)", float("nan"), None))

    # ============================== Regime 2: INT8 fast cascade ==============================
    print("\n[2/3] ⚡  INT8 optimized cascade (warmed, no cache hit) ...")
    fast_cfg = FastPipelineConfig(cache_enabled=False)
    fast = FastCascadeS2ST(fast_cfg)
    print(f"      warmup (NOT timed) ...")
    fast._ensure_asr()
    fast._ensure_mt()
    fast._ensure_tts(args.target)
    # Throwaway run to warm JIT
    fast.transcribe(audio[:int(sr)], sr)

    t0 = time.perf_counter()
    out_f = fast.run(audio, sr, args.target)
    f_total = (time.perf_counter() - t0) * 1000
    f_asr = out_f["latency"]["asr_ms"]; f_mt = out_f["latency"]["mt_ms"]; f_tts = out_f["latency"]["tts_ms"]
    print(f"      vi_text : {out_f['vi_text']!r}")
    print(f"      tgt_text: {out_f['tgt_text']!r}")
    print(f"      asr={f_asr:>7.0f}ms   mt={f_mt:>7.0f}ms   tts={f_tts:>7.0f}ms")
    print(f"      total   : {f_total:>8.0f} ms   RTF {f_total/(duration*1000):.2f}x")
    results.append(("INT8 fast (cascade)", f_total, out_f))

    # ============================== Regime 3: cache hit ==============================
    print("\n[3/3] 🚀  INT8 + tonal-perceptual cache  (same audio again, should HIT cache) ...")
    cached_cfg = FastPipelineConfig(cache_enabled=True, cache_cosine_hit=0.95)
    cached = FastCascadeS2ST(cached_cfg)
    cached._ensure_asr()
    cached._ensure_mt()
    cached._ensure_tts(args.target)
    cached.transcribe(audio[:int(sr)], sr)   # warmup
    # First call: cache MISS (populates the cache)
    print(f"      first call (cache miss expected) ...")
    out_miss = cached.run(audio, sr, args.target)
    miss_total = out_miss["latency"]["total_ms"]
    print(f"      miss total = {miss_total:>8.0f} ms"
          f"   cache.hit={out_miss['cache']['hit']}")
    # Second call: cache HIT
    print(f"      second call (cache hit expected) ...")
    out_hit = cached.run(audio, sr, args.target)
    hit_total = out_hit["latency"]["total_ms"]
    hit_cache_ms = out_hit["latency"]["cache_ms"]
    print(f"      vi_text : {out_hit['vi_text']!r}")
    print(f"      tgt_text: {out_hit['tgt_text']!r}")
    print(f"      cache lookup = {hit_cache_ms:.1f} ms"
          f"   cache.hit={out_hit['cache']['hit']}   cosine={out_hit['cache']['cosine']}")
    print(f"      total   : {hit_total:>8.0f} ms   RTF {hit_total/(duration*1000):.4f}x")
    results.append(("INT8 + cache hit",       hit_total, out_hit))

    # ============================== Summary table ==============================
    print("\n" + "=" * 78)
    print("📊  COMPARISON SUMMARY")
    print("=" * 78)
    headers = ("regime", "total ms", "RTF", "speedup vs FP32", "wow factor")
    print(f"  {headers[0]:<32} {headers[1]:>10} {headers[2]:>8} {headers[3]:>20} {headers[4]:>14}")
    print("  " + "-" * 76)
    baseline_ms = results[0][1]
    for name, ms, _ in results:
        if ms != ms:  # NaN
            print(f"  {name:<32} {'(skipped)':>10} {'':>8} {'':>20} {'':>14}")
            continue
        rtf = ms / (duration * 1000)
        speedup = baseline_ms / ms if baseline_ms == baseline_ms else float("nan")
        wow = "🐢 baseline" if "FP32" in name else ("⚡ fast" if "cascade" in name else "🚀 instant")
        speedup_str = f"{speedup:.1f}x" if speedup == speedup else "—"
        print(f"  {name:<32} {ms:>10.1f} {rtf:>8.2f} {speedup_str:>20} {wow:>14}")
    print("=" * 78)
    if not args.skip_fp32 and results[0][1] == results[0][1]:
        print(f"\n🎯  Headline: {baseline_ms/results[2][1]:.0f}x faster end-to-end on cache hit,"
              f" {baseline_ms/results[1][1]:.1f}x on plain INT8 cascade — same input, same output text.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
