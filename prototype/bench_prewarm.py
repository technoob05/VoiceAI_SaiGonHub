"""
Benchmark warm-path optimizations on FastCascadeS2ST.

Spawns each variant as a separate subprocess so cold-Python state and HF cache
warmth match across runs. Writes Results/prewarm_bench.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


SUBPROC = r'''
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path(r"{HERE}")
sys.path.insert(0, r"{HERE}")
import numpy as np
from pipeline_fast import FastCascadeS2ST, FastPipelineConfig

mode = sys.argv[1]  # "off" or "on"

t0 = time.perf_counter()
pipe = FastCascadeS2ST(FastPipelineConfig())

# Monkey-patch off prewarm/pretokenize for "off" mode.
if mode == "off":
    pipe._asr_prewarm = lambda: None
    pipe._pretokenize_industrial_phrases = lambda: None

pipe._ensure_asr()
asr_prewarm_ms = pipe._asr_prewarm_ms or 0.0
pipe._ensure_mt()
init_ms = (time.perf_counter() - t0) * 1000

phrases_path = Path(r"{HERE}") / "industrial_phrases.json"
raw = json.loads(phrases_path.read_text(encoding="utf-8"))
vi_phrases = []
for cat, items in raw.items():
    if cat.startswith("_"): continue
    for it in items: vi_phrases.append(it["vi"])

t = time.perf_counter()
_ = pipe.translate(vi_phrases[0], "en", source="vi")
first_ms = (time.perf_counter() - t) * 1000

per_call = []
for i in range(1, 11):
    t = time.perf_counter()
    _ = pipe.translate(vi_phrases[i % len(vi_phrases)], "en", source="vi")
    per_call.append((time.perf_counter() - t) * 1000)
steady_ms = float(np.median(per_call[-5:]))

out = {{
    "mode": mode,
    "init_ms": round(init_ms, 1),
    "asr_prewarm_ms": round(asr_prewarm_ms, 1),
    "first_call_ms": round(first_ms, 1),
    "steady_ms": round(steady_ms, 1),
    "per_call_ms": [round(x, 1) for x in per_call],
}}
print("__BENCH_JSON__" + json.dumps(out))
'''.format(HERE=str(HERE).replace("\\", "\\\\"))


def run_variant(mode: str) -> dict:
    script = HERE / f"_bench_subproc_{mode}.py"
    script.write_text(SUBPROC, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    script.unlink(missing_ok=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(out)
    for line in out.splitlines():
        if line.startswith("__BENCH_JSON__"):
            return json.loads(line[len("__BENCH_JSON__"):])
    raise RuntimeError(f"no result from subprocess (mode={mode})")


def main() -> int:
    # Order matters: run "off" first (cold filesystem cache) so the comparison
    # to "on" doesn't get a free HF-cache + page-cache speedup. We also do a
    # warm-up dummy import to neutralize OS page cache across both runs.
    print("=== warmup pass (discard) ===")
    _ = run_variant("off")
    print("=== Variant 1: WITHOUT prewarm + pre-tokenize ===")
    off = run_variant("off")
    print("=== Variant 2: WITH prewarm + pre-tokenize ===")
    on = run_variant("on")

    report = {
        "baseline_off": off,
        "with_warmups": on,
        "delta": {
            "init_ms_extra":     round(on["init_ms"] - off["init_ms"], 1),
            "first_call_saved":  round(off["first_call_ms"] - on["first_call_ms"], 1),
            "steady_state_diff": round(on["steady_ms"] - off["steady_ms"], 1),
            "asr_prewarm_cost":  on["asr_prewarm_ms"],
        },
    }
    out_path = PROJECT / "Results" / "prewarm_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"[write] {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
