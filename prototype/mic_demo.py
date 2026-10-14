"""mic_demo.py — dead-simple NATIVE microphone translator (no browser, no ffmpeg).

The Gradio web demo records the mic through the browser (webm/opus → needs ffmpeg,
needs browser mic permission, fragile). This script captures straight from the OS
microphone via `sounddevice` — the path that just verified working (RMS>0) on this
machine — so it Just Works with your real mic.

Flow (loops): press ENTER → speak → press ENTER again to stop → see the Vietnamese
transcript, the translation, and hear the synthesized target speech. Ctrl-C to quit.

    python mic_demo.py --target en          # Vietnamese → English
    python mic_demo.py --target zh --punct  # + punctuation/capitalisation
    python mic_demo.py --seconds 5          # fixed 5s capture instead of press-to-stop
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

SR = 16_000


def record_until_enter() -> np.ndarray:
    import sounddevice as sd
    frames: list[np.ndarray] = []
    stop = threading.Event()

    def cb(indata, n, t, status):
        frames.append(indata.copy())

    input("  ▶ Press ENTER to START recording…")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        print("  ● Recording… speak now, then press ENTER to STOP.", flush=True)
        input()
    stop.set()
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames).reshape(-1)


def record_fixed(seconds: float) -> np.ndarray:
    import sounddevice as sd
    for i in range(3, 0, -1):
        print(f"  recording in {i}…", flush=True); time.sleep(1)
    print(f"  ● Recording {seconds:.0f}s… speak now.", flush=True)
    a = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return a.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="en", choices=["en", "zh", "ko", "vi"])
    ap.add_argument("--seconds", type=float, default=0, help="fixed-length capture (0 = press-to-stop)")
    ap.add_argument("--punct", action="store_true", help="restore Vietnamese punctuation/caps")
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args()

    print("Loading models (one-time ~30-60s)…", flush=True)
    import os
    os.environ.setdefault("VOICEAI_TTS_BACKEND", "piper")
    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
    cfg = FastPipelineConfig()
    cfg.punct = bool(args.punct)
    pipe = FastCascadeS2ST(cfg)
    pipe._ensure_asr(); pipe._ensure_mt(); pipe.synthesize("khởi động", args.target)
    try:
        import sounddevice as sd
    except Exception:
        sd = None
    print(f"Ready. Target = {args.target}.  (Ctrl-C to quit)\n", flush=True)

    while True:
        try:
            audio = record_fixed(args.seconds) if args.seconds > 0 else record_until_enter()
            if len(audio) < SR // 2:
                print("  (too short / nothing captured — try again)\n"); continue
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"  captured {len(audio)/SR:.1f}s (RMS={rms:.4f}) → translating…", flush=True)
            t0 = time.perf_counter()
            res = pipe.run(audio, SR, args.target)
            dt = (time.perf_counter() - t0) * 1000
            print(f"  🇻🇳 {res['vi_text']}")
            print(f"  🌐 {res['tgt_text']}   ({dt:.0f} ms)")
            if not args.no_play and sd is not None:
                a = res["tgt_audio"]
                if a is not None and len(a):
                    sd.play(a, res["sr"]); sd.wait()
            print()
        except KeyboardInterrupt:
            print("\nbye."); break


if __name__ == "__main__":
    main()
