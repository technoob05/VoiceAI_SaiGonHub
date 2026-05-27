"""
Pre-warm the translation memory (Tier-2 text cache) with curated industrial
Vietnamese phrases.

This is the **demo killer feature**: a judge speaks any of these common phrases
(safety, production-line, communication, schedule, common objects) and hits the
cache on the very first try — even though the judge never demoed the system
before. Effective latency drops to TTS-only (~1-3 s on CPU) for any pre-warmed
phrase, vs full cascade (~5-7 s).

Run programmatically inside the web demo via the "🔥 Pre-warm cache" button, or
standalone via CLI:

    python prototype/preload_cache.py --target en
    python prototype/preload_cache.py --target en --include-audio   # also synth TTS so T1 audio cache hits
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

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

PHRASES_FILE = HERE / "industrial_phrases.json"


def load_phrases() -> list[dict]:
    data = json.loads(PHRASES_FILE.read_text(encoding="utf-8"))
    flat = []
    for category, items in data.items():
        if category.startswith("_"):
            continue
        for item in items:
            flat.append({"category": category, **item})
    return flat


def warm_translation_memory(pipe, target: str = "en",
                            include_audio: bool = False,
                            verbose: bool = True) -> dict:
    """Populate `pipe._tm` (and optionally synthesize TTS audio for T1 cache too).

    Returns summary {n_phrases, n_inserted, total_ms}.
    """
    phrases = load_phrases()
    if pipe._tm is None:
        return {"error": "translation memory not enabled on this pipeline"}

    t0 = time.perf_counter()
    n_inserted = 0
    for i, p in enumerate(phrases, start=1):
        vi  = p["vi"]
        tgt = p.get(target)
        if not tgt:
            continue   # no translation available for this target lang
        audio = None
        if include_audio and pipe is not None:
            try:
                wav, sr = pipe.synthesize(tgt, target)
                audio = (sr, wav)
            except Exception as e:
                if verbose:
                    print(f"  [warn] TTS failed for {tgt!r}: {e}")
        pipe._tm.put(vi, tgt, target, audio=audio)
        n_inserted += 1
        if verbose and i % 10 == 0:
            print(f"  warmed {i}/{len(phrases)}: {p['category']}/{vi!r} -> {tgt!r}")

    total_ms = (time.perf_counter() - t0) * 1000
    return {"n_phrases": len(phrases), "n_inserted": n_inserted,
            "target": target, "include_audio": include_audio,
            "total_ms": round(total_ms, 1),
            "tm_stats": pipe._tm.summary()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",        default="en", choices=["en", "zh", "ko"])
    ap.add_argument("--include-audio", action="store_true",
                    help="Also synthesize TTS for each phrase (slower, but enables T1 audio-cache hits)")
    args = ap.parse_args()

    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
    cfg = FastPipelineConfig()
    pipe = FastCascadeS2ST(cfg)
    pipe._ensure_mt()
    if args.include_audio:
        pipe._ensure_tts(args.target)

    summary = warm_translation_memory(pipe, args.target, include_audio=args.include_audio)
    print()
    print("=" * 60)
    print("PRE-WARM SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<14}: {v}")
    print("=" * 60)
    print("Now any of these phrases will hit the cache on first try in the web demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
