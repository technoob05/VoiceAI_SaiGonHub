"""Interleaved A/B latency bench for the offline TTS voices — Mandarin focus.

The T4 bidirectional run (Results/kaggle_t4/README.md section C) put Mandarin TTS
at 877 ms p50 against 149-470 ms for the other three voices. This script isolates
that stage on a single machine so the cause can be attributed rather than guessed:

  cost      per-voice ms and ms per second of *output audio* (the fair unit —
            MeloTTS emits 44.1 kHz, every other voice emits 22.05 kHz)
  shape     latency vs output length, to separate a fixed per-call cost
            (jieba/lexicon/rule-FST text processing) from per-sample model cost
  threads   num_threads sweep for one voice
  ab        two configurations interleaved call-by-call so machine drift cancels
  load      engine build wall clock: serial first-touch vs PiperTTS.prewarm()

Usage (models dir must be absolute):
  VOICEAI_PIPER_MODELS_DIR=.../data/raw/piper-models \
  python exp/bench_tts_mandarin.py cost --langs zh,en,vi,ko --reps 3
  python exp/bench_tts_mandarin.py ab --lang zh --env-a VOICEAI_TTS_THREADS_ZH=4 \
                                      --env-b VOICEAI_TTS_THREADS_ZH=6 --reps 3
  python exp/bench_tts_mandarin.py load --langs zh,en,vi,ko
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototype"))

# Real decoder output from the T4 bidirectional run, so the workload is the one
# the device actually synthesizes rather than a hand-written sentence.
TEXTS = {
    "zh": ["遗址有七个,两座是木材.",
           "他们在人群中无畏地放荡.",
           "我从来没有过奶.",
           "造成了中东地区的暴力事件,",
           "非常有创意.",
           "区域B的空气压缩机正在过热,请停止生产线.",
           "叉车正在向后移动.",
           "货物需要在仓库发送前检查."],
    "en": ["The relic has seven heaps and two sheaps made of metric wood.",
           "You are comfortable getting dirty in a crowded place without fear.",
           "The air compressor in zone B is overheating, stop the production line.",
           "The forklift is reversing.",
           "The shipment must be checked before dispatch."],
    "vi": ["Di tích có bảy đống và hai đống làm bằng gỗ.",
           "Máy nén khí ở khu B đang quá nóng, hãy dừng dây chuyền.",
           "Xe nâng đang lùi.",
           "Lô hàng cần kiểm tra trước khi xuất kho."],
    "ko": ["유적지에는 일곱 개의 더미가 있습니다.",
           "B구역 공기 압축기가 과열되고 있습니다.",
           "지게차가 후진하고 있습니다.",
           "출고 전에 화물을 확인해야 합니다."],
}


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _tts(**kw):
    from local_tts_piper import PiperTTS
    return PiperTTS(auto_download=False, **kw)


def _time_one(tts, text, lang):
    t0 = time.perf_counter()
    wav, sr = tts.synthesize(text, lang)
    return (time.perf_counter() - t0) * 1000.0, len(wav) / sr, sr


def cmd_cost(args):
    langs = args.langs.split(",")
    tts = _tts(cache_cap=len(langs))
    out = {}
    for lang in langs:
        tts.synthesize(TEXTS[lang][0], lang)          # warm engine, discard
        lat, aud, sr = [], [], 0
        for _ in range(args.reps):
            for t in TEXTS[lang]:
                ms, s, sr = _time_one(tts, t, lang)
                lat.append(ms)
                aud.append(s)
        out[lang] = {"n": len(lat), "sample_rate": sr,
                     "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95),
                     "mean_ms": statistics.mean(lat), "audio_p50_s": pct(aud, .5),
                     "ms_per_audio_s": sum(lat) / sum(aud)}
        r = out[lang]
        print(f"{lang:3s} n={r['n']:3d} sr={sr:6d}  p50={r['p50_ms']:7.1f}ms  "
              f"p95={r['p95_ms']:7.1f}ms  ms/audio-s={r['ms_per_audio_s']:7.1f}", flush=True)
    return out


def cmd_shape(args):
    """Latency vs output length. A large intercept would mean text processing
    (jieba/lexicon/rule-FST) dominates; a pure slope means the VITS forward does."""
    tts = _tts(cache_cap=1)
    lang = args.lang
    probes = {"zh": ["好", "你好世界", TEXTS["zh"][4], TEXTS["zh"][0],
                     "".join(TEXTS["zh"][:4])],
              "en": ["Go.", "Stop the line.", TEXTS["en"][3], TEXTS["en"][0],
                     " ".join(TEXTS["en"][:3])]}[lang]
    tts.synthesize(probes[0], lang)
    rows = []
    for p in probes:
        lat, aud = [], []
        for _ in range(args.reps):
            ms, s, _ = _time_one(tts, p, lang)
            lat.append(ms)
            aud.append(s)
        rows.append({"chars": len(p), "audio_s": statistics.median(aud),
                     "p50_ms": statistics.median(lat)})
        print(f"  chars={len(p):4d} audio={rows[-1]['audio_s']:5.2f}s "
              f"p50={rows[-1]['p50_ms']:8.1f}ms", flush=True)
    xs = [r["audio_s"] for r in rows]
    ys = [r["p50_ms"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    print(f"  fit: {slope:.0f} ms per second of audio + {my - slope * mx:.0f} ms fixed")
    return {"rows": rows, "slope_ms_per_audio_s": slope, "intercept_ms": my - slope * mx}


def cmd_threads(args):
    lang = args.lang
    out = {}
    for nt in [int(x) for x in args.threads.split(",")]:
        tts = _tts(num_threads=nt, cache_cap=1)
        t0 = time.perf_counter()
        tts.synthesize(TEXTS[lang][0], lang)
        build = (time.perf_counter() - t0) * 1000
        lat, aud = [], []
        for _ in range(args.reps):
            for t in TEXTS[lang]:
                ms, s, _ = _time_one(tts, t, lang)
                lat.append(ms)
                aud.append(s)
        out[nt] = {"p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95),
                   "ms_per_audio_s": sum(lat) / sum(aud), "build_ms": build}
        print(f"  threads={nt}: p50={out[nt]['p50_ms']:7.1f}ms  p95={out[nt]['p95_ms']:7.1f}ms  "
              f"ms/audio-s={out[nt]['ms_per_audio_s']:7.1f}  build={build:6.0f}ms", flush=True)
        del tts
    return out


def _apply(envs: str):
    for kv in envs.split(";"):
        if kv.strip():
            k, _, v = kv.partition("=")
            os.environ[k.strip()] = v.strip()


def cmd_ab(args):
    """Two configurations, interleaved call by call, so thermal/scheduler drift
    hits both arms equally. Each arm keeps its own engine, both stay resident."""
    lang = args.lang
    arms = {}
    for name, envs in (("A", args.env_a), ("B", args.env_b)):
        _apply(envs)
        t0 = time.perf_counter()
        tts = _tts(cache_cap=1)
        tts.synthesize(TEXTS[lang][0], lang)
        arms[name] = {"tts": tts, "env": envs, "lat": [], "aud": [],
                      "build_ms": (time.perf_counter() - t0) * 1000}
        print(f"arm {name} [{envs}] warm+build {arms[name]['build_ms']:.0f}ms", flush=True)
    texts = TEXTS[lang] * args.reps
    for i, t in enumerate(texts):
        for name in (["A", "B"] if i % 2 == 0 else ["B", "A"]):
            ms, s, _ = _time_one(arms[name]["tts"], t, lang)
            arms[name]["lat"].append(ms)
            arms[name]["aud"].append(s)
    out = {}
    for name, a in arms.items():
        out[name] = {"env": a["env"], "n": len(a["lat"]), "build_ms": a["build_ms"],
                     "p50_ms": pct(a["lat"], .5), "p95_ms": pct(a["lat"], .95),
                     "mean_ms": statistics.mean(a["lat"]),
                     "ms_per_audio_s": sum(a["lat"]) / sum(a["aud"])}
        r = out[name]
        print(f"arm {name} [{r['env']}] n={r['n']} p50={r['p50_ms']:7.1f}ms "
              f"p95={r['p95_ms']:7.1f}ms ms/audio-s={r['ms_per_audio_s']:7.1f}")
    if "A" in out and "B" in out:
        print(f"B/A p50 = {out['B']['p50_ms'] / out['A']['p50_ms']:.3f}   "
              f"B/A p95 = {out['B']['p95_ms'] / out['A']['p95_ms']:.3f}")
    return out


def cmd_load(args):
    """Voice-load wall clock: serial first-touch vs parallel prewarm()."""
    langs = args.langs.split(",")
    out = {}
    t0 = time.perf_counter()
    serial = _tts(cache_cap=len(langs))
    per = {}
    for lang in langs:
        t1 = time.perf_counter()
        serial.synthesize(".", lang)
        per[lang] = (time.perf_counter() - t1) * 1000
    out["serial_ms"] = (time.perf_counter() - t0) * 1000
    out["serial_per_voice_ms"] = per
    print("serial:", {k: round(v) for k, v in per.items()},
          f"total={out['serial_ms']:.0f}ms", flush=True)
    del serial
    par = _tts(cache_cap=len(langs))
    t0 = time.perf_counter()
    par.prewarm(langs)
    out["prewarm_ms"] = (time.perf_counter() - t0) * 1000
    print(f"prewarm (parallel): total={out['prewarm_ms']:.0f}ms  "
          f"speedup={out['serial_ms'] / out['prewarm_ms']:.2f}x")
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost")
    c.add_argument("--langs", default="zh,en,vi,ko")
    c.add_argument("--reps", type=int, default=3)
    s = sub.add_parser("shape")
    s.add_argument("--lang", default="zh")
    s.add_argument("--reps", type=int, default=5)
    t = sub.add_parser("threads")
    t.add_argument("--lang", default="zh")
    t.add_argument("--threads", default="1,2,4,6,8")
    t.add_argument("--reps", type=int, default=2)
    a = sub.add_parser("ab")
    a.add_argument("--lang", default="zh")
    a.add_argument("--env-a", default="")
    a.add_argument("--env-b", default="")
    a.add_argument("--reps", type=int, default=3)
    l = sub.add_parser("load")
    l.add_argument("--langs", default="zh,en,vi,ko")
    for p in (c, s, t, a, l):
        p.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {"cost": cmd_cost, "shape": cmd_shape, "threads": cmd_threads,
           "ab": cmd_ab, "load": cmd_load}[args.cmd](args)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
