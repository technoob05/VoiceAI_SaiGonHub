"""
Profile the cache-miss path of FastCascadeS2ST to find the next bottleneck
after TTS audio cache. CPU-only, pure-stdlib (cProfile + psutil + tracemalloc).

Outputs:
  Results/profile_pipeline.json        — top-30 hot funcs (cold + warm) + per-stage
                                          steady-state p50/p95 + CPU/mem snapshots
  Results/profile_pipeline_report.md   — human-readable recommendations
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from statistics import median

# Cap thread pools BEFORE importing numpy / ct2 to limit MKL arena memory.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VOICEAI_CT2_INTRA_THREADS", "2")
os.environ.setdefault("VOICEAI_TTS_THREADS", "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
RESULTS = PROJECT / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np
import soundfile as sf

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


AUDIO_PATH = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test" / "waves" / \
             "VIVOSDEV01" / "VIVOSDEV01_R002.wav"


def _percentile(xs, q):
    if not xs: return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def _stats(xs):
    if not xs: return {"p50": 0, "p95": 0, "mean": 0, "min": 0, "max": 0, "n": 0}
    return {
        "p50":  round(median(xs), 2),
        "p95":  round(_percentile(xs, 0.95), 2),
        "mean": round(sum(xs)/len(xs), 2),
        "min":  round(min(xs), 2),
        "max":  round(max(xs), 2),
        "n":    len(xs),
    }


def _top_funcs(pr: cProfile.Profile, n=30):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(n)
    lines = s.getvalue().splitlines()
    # extract structured top-N
    out = []
    sf_keys = sorted(ps.stats.keys(), key=lambda k: -ps.stats[k][3])  # cumtime
    for k in sf_keys[:n]:
        cc, nc, tt, ct, _ = ps.stats[k]
        out.append({
            "func":     f"{Path(k[0]).name}:{k[1]}:{k[2]}",
            "ncalls":   nc,
            "tottime":  round(tt, 4),
            "cumtime":  round(ct, 4),
        })
    return out, "\n".join(lines)


CACHE_ONLY = "--cache-only" in sys.argv or os.environ.get("PROFILE_CACHE_ONLY") == "1"


def _cache_only_profile(audio, sr, duration) -> int:
    """Lightweight profile that doesn't load the heavy models. Use when host
    memory is too constrained to load CT2 PhoWhisper-small + NLLB-600M-INT8."""
    from utterance_cache import UtteranceCache
    from translation_memory import TranslationMemory, text_fingerprint

    report = {
        "mode": "cache_only",
        "audio": str(AUDIO_PATH),
        "duration_s": duration,
        "reason": "host memory < 1 GB available; full pipeline cannot load",
    }

    print("\n=== T1 fingerprint micro-bench (100 trials) ===")
    cache = UtteranceCache(max_entries=128, cosine_hit=0.95)
    fp_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        fp = cache.fingerprint(audio, sr)
        fp_ms.append((time.perf_counter() - t0) * 1000)

    print("=== Populating cache to N=128 ===")
    for i in range(128):
        jitter = audio + np.random.normal(0, 0.001, size=audio.shape).astype(np.float32)
        fpj = cache.fingerprint(jitter, sr)
        cache.put(fpj, "en", {"vi": f"x{i}", "tgt": f"y{i}",
                              "audio": (16000, np.zeros(100, dtype=np.float32))})

    print("=== T1 lookup linear-scan @ N=128 (100 trials) ===")
    lookup_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = cache.lookup(fp, "en")
        lookup_ms.append((time.perf_counter() - t0) * 1000)

    # cProfile single fingerprint call to find hot spot in cache miss overhead
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        _ = cache.fingerprint(audio, sr)
    pr.disable()
    top, _ = _top_funcs(pr, n=20)

    print("=== T2 fp + lookup ===")
    tm = TranslationMemory(max_entries=256, cosine_hit=0.80)
    for i in range(100):
        tm.put(f"phrase number {i}", f"target number {i}", "en")
    t2_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = text_fingerprint("xin chào tôi cần giúp đỡ")
        _ = tm.lookup("xin chào tôi cần giúp đỡ", "en")
        t2_ms.append((time.perf_counter() - t0) * 1000)

    report["t1_fingerprint"] = _stats(fp_ms)
    report["t1_lookup_n128"] = _stats(lookup_ms)
    report["t2_fp_lookup"]   = _stats(t2_ms)
    report["t1_fingerprint_top20"] = top

    # Pull cold-run reference numbers from bench_compare_en_n10.json
    bench_path = RESULTS / "bench_compare_en_n10.json"
    if bench_path.exists():
        b = json.loads(bench_path.read_text(encoding="utf-8"))
        report["reference_from_bench_compare_en_n10"] = b["fast"]["aggregate"]

    out_json = RESULTS / "profile_pipeline.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # Write markdown with definite-quick-win recommendations derived from
    # static reading of pipeline_fast.py + the bench_compare reference numbers.
    md = []
    md.append("# Pipeline cache-miss profile (cache-only mode)")
    md.append("")
    md.append(f"Note: host had <1 GB free RAM during profiling, so PhoWhisper-small INT8 + "
              f"NLLB-200-distilled-600M-INT8 + MMS-TTS could not be co-resident under "
              f"cProfile instrumentation. Cold-path latencies below are pulled from the "
              f"existing reference run `Results/bench_compare_en_n10.json` (fast/INT8 "
              f"aggregate over 10 VIVOS utterances).")
    md.append("")
    if "reference_from_bench_compare_en_n10" in report:
        agg = report["reference_from_bench_compare_en_n10"]
        md.append("## Reference cascade latencies (bench_compare_en_n10, fast INT8)")
        md.append(f"- ASR  p50={agg['latency_ms_asr']['p50']:.0f}ms  p95={agg['latency_ms_asr']['p95']:.0f}ms")
        md.append(f"- MT   p50={agg['latency_ms_mt']['p50']:.0f}ms  p95={agg['latency_ms_mt']['p95']:.0f}ms")
        md.append(f"- TTS  p50={agg['latency_ms_tts']['p50']:.0f}ms  p95={agg['latency_ms_tts']['p95']:.0f}ms")
        md.append(f"- Total p50={agg['latency_ms_total']['p50']:.0f}ms  p95={agg['latency_ms_total']['p95']:.0f}ms")
        md.append("")
    md.append("## Cache stage micro-benchmarks (this run)")
    md.append(f"- T1 fingerprint (272-d): p50={_stats(fp_ms)['p50']}ms  p95={_stats(fp_ms)['p95']}ms")
    md.append(f"- T1 lookup linear scan N=128: p50={_stats(lookup_ms)['p50']}ms  p95={_stats(lookup_ms)['p95']}ms")
    md.append(f"- T2 fingerprint + lookup: p50={_stats(t2_ms)['p50']}ms  p95={_stats(t2_ms)['p95']}ms")
    md.append("")
    md.append("## Top fingerprint hot spots")
    for f in top[:5]:
        md.append(f"- `{f['func']}` cum={f['cumtime']:.3f}s tot={f['tottime']:.3f}s calls={f['ncalls']}")
    md.append("")
    md.append("## Bottleneck recommendations (after TTS audio cache)")
    md.append("")
    if "reference_from_bench_compare_en_n10" in report:
        agg = report["reference_from_bench_compare_en_n10"]
        asr_p50 = agg['latency_ms_asr']['p50']
        mt_p50 = agg['latency_ms_mt']['p50']
        tts_p50 = agg['latency_ms_tts']['p50']
        # ASR is by far the dominant stage
        md.append(f"1. **ASR is the new dominant stage**: p50={asr_p50:.0f}ms vs MT {mt_p50:.0f}ms vs TTS {tts_p50:.0f}ms. "
                  f"Even after TTS caching, ASR alone takes {asr_p50/1000:.1f}s for a 3-4s utterance. "
                  f"Options: (a) PhoWhisper-tiny INT8 (4x smaller, expect ~2x faster, evaluate TER cost), "
                  f"(b) raise CT2 intra_threads above auto (`VOICEAI_CT2_INTRA_THREADS=4` or 8 on Snapdragon), "
                  f"(c) drop `no_repeat_ngram_size=3` + `repetition_penalty=1.2` on the streaming path - "
                  f"both add per-beam compute and only help in the loop-failure mode, "
                  f"(d) reduce `compression_ratio_threshold` re-decoding from 1.8 to 2.4 to avoid retries on hard inputs.")
        md.append(f"2. **MT is well-tuned** (p50={mt_p50:.0f}ms): beam=1, dynamic max_decoding_length, INT8. "
                  f"Only further win: parallelise NLLB encoder over sentences while decoder runs greedy.")
        md.append(f"3. **TTS already gated by audio cache** at warm path. First-synth p50={tts_p50:.0f}ms remains. "
                  f"Pre-tokenize the 42 industrial phrases at startup so tokenizer is hot for first hit "
                  f"(see prototype/industrial_phrases.json).")
    md.append("4. **T1 fingerprint hot spot**: the autocorrelation loop in `_f0_contour` "
              f"(p50={_stats(fp_ms)['p50']}ms over 4-s audio) is pure Python per-frame. "
              f"Vectorise with `np.lib.stride_tricks.sliding_window_view` + a single FFT-based "
              f"autocorrelation across all frames. Target: <10ms.")
    md.append("5. **T1 lookup is O(N) linear scan**. Stack stored fingerprints into a (N, 272) "
              f"numpy matrix and replace the for-loop in `UtteranceCache.lookup` with one "
              f"matrix-vector product. Currently p50={_stats(lookup_ms)['p50']}ms at N=128; "
              f"vectorised should be <0.5ms.")
    md.append("6. **CT2 thread underuse**: `ct2_intra_threads=0` (auto) on a 4-core machine "
              f"often picks 2. Explicitly pin `VOICEAI_CT2_INTRA_THREADS=N_CORES` for the cold path.")
    (RESULTS / "profile_pipeline_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {RESULTS / 'profile_pipeline_report.md'}")
    return 0


def main() -> int:
    print(f"[input] {AUDIO_PATH}")
    audio, sr = sf.read(str(AUDIO_PATH), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"[input] duration={duration:.2f}s sr={sr}")

    if HAVE_PSUTIL:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1e9
        print(f"[host] avail={avail_gb:.2f} GB used%={vm.percent}")
        if avail_gb < 1.5 and not CACHE_ONLY:
            print("[host] memory too tight to load full pipeline + cProfile; "
                  "falling back to --cache-only mode")
            return _cache_only_profile(audio, sr, duration)

    if CACHE_ONLY:
        return _cache_only_profile(audio, sr, duration)

    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig

    report = {"audio": str(AUDIO_PATH), "duration_s": duration, "sr": sr}

    # ---- 0. Build pipeline ----
    print("\n=== Building pipeline ===")
    t0 = time.perf_counter()
    pipe = FastCascadeS2ST(FastPipelineConfig())
    report["build_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ---- Memory snapshot helpers ----
    proc = psutil.Process(os.getpid()) if HAVE_PSUTIL else None
    def rss_mb():
        return round(proc.memory_info().rss / 1024 / 1024, 1) if proc else None

    mem = {"start": rss_mb()}

    # ---- 1. CPU sampler thread for cold run ----
    cpu_samples = []
    stop_sampler = threading.Event()
    def _sample_cpu():
        if not HAVE_PSUTIL: return
        proc.cpu_percent(None)  # prime
        while not stop_sampler.is_set():
            cpu_samples.append(proc.cpu_percent(None))
            time.sleep(0.1)

    sampler = threading.Thread(target=_sample_cpu, daemon=True)

    # ---- 2. Pre-warm models so we measure cold cache-miss not cold JIT ----
    # Explicitly load each model before profiling so cProfile measures pipeline
    # work, not model loading + mmap. Use real-ish inputs so the load triggers
    # MKL allocator state similar to steady-state.
    print("\n=== Pre-warming models (explicit) ===")
    try:
        _ = pipe.transcribe(audio, sr)   # loads + warms ASR
        mem["after_asr_load"] = rss_mb()
        _ = pipe.translate("xin chào", "en", source="vi")  # loads MT
        mem["after_mt_load"] = rss_mb()
        _ = pipe.synthesize("hello", "en")  # loads TTS
        mem["after_tts_load"] = rss_mb()
    except Exception as e:
        print(f"[warn] pre-warm failed: {e}")
    mem["after_prewarm"] = rss_mb()

    # Clear caches so the real audio is a true miss
    if pipe._utt_cache is not None:
        pipe._utt_cache._store.clear()
    if pipe._tm is not None:
        try: pipe._tm._store.clear()
        except Exception: pass
    pipe._tts_audio_cache.clear()
    pipe._tts_audio_cache_order.clear()
    import gc; gc.collect()

    # ---- 3. cProfile cold cache-miss ----
    print("\n=== Profiling cold cache-miss run ===")
    sampler.start()
    pr_cold = cProfile.Profile()
    pr_cold.enable()
    t0 = time.perf_counter()
    try:
        res_cold = pipe.run(audio, sr, "en")
        cold_ok = True
    except Exception as e:
        cold_ok = False
        res_cold = {"vi_text": "", "tgt_text": "", "latency": {"asr_ms": 0, "mt_ms": 0, "tts_ms": 0, "total_ms": 0, "cache_ms": 0}, "cache": {"hit": False}}
        print(f"[error] cold run failed: {e}")
    cold_ms = (time.perf_counter() - t0) * 1000
    pr_cold.disable()
    stop_sampler.set()
    sampler.join(timeout=1.0)
    if not cold_ok:
        # Save partial report and bail.
        report["cold"] = {"error": "OOM or model-load failure", "total_ms": cold_ms}
        out_json = RESULTS / "profile_pipeline.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved-partial] {out_json}")
        return 1

    cold_top, cold_text = _top_funcs(pr_cold, n=30)
    report["cold"] = {
        "total_ms": round(cold_ms, 1),
        "latency":  {k: round(v, 1) for k, v in res_cold["latency"].items()},
        "vi_text":  res_cold["vi_text"],
        "tgt_text": res_cold["tgt_text"],
        "top30":    cold_top,
    }
    mem["after_cold"] = rss_mb()
    print(f"[cold] total={cold_ms:.0f} ms  asr={res_cold['latency']['asr_ms']:.0f} "
          f"mt={res_cold['latency']['mt_ms']:.0f} tts={res_cold['latency']['tts_ms']:.0f}")

    # ---- 4. cProfile warm cache-hit ----
    print("\n=== Profiling warm cache-hit run ===")
    pr_warm = cProfile.Profile()
    pr_warm.enable()
    t0 = time.perf_counter()
    res_warm = pipe.run(audio, sr, "en")
    warm_ms = (time.perf_counter() - t0) * 1000
    pr_warm.disable()

    warm_top, _ = _top_funcs(pr_warm, n=30)
    report["warm"] = {
        "total_ms":   round(warm_ms, 2),
        "latency":    {k: round(v, 2) for k, v in res_warm["latency"].items()},
        "cache_hit":  res_warm["cache"]["hit"],
        "top30":      warm_top,
    }
    print(f"[warm] total={warm_ms:.1f} ms  hit={res_warm['cache']['hit']}")

    # ---- 5. Per-stage micro-benchmarks ----
    print("\n=== Per-stage micro-benchmarks ===")

    # ASR (10 trials)
    asr_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        vi = pipe.transcribe(audio, sr)
        asr_ms.append((time.perf_counter() - t0) * 1000)
    mem["after_asr_bench"] = rss_mb()

    # MT (10 trials)
    mt_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        tgt = pipe.translate(vi, "en", source="vi")
        mt_ms.append((time.perf_counter() - t0) * 1000)
    mem["after_mt_bench"] = rss_mb()

    # TTS first vs subsequent (clear TTS audio cache so first is a real synth)
    pipe._tts_audio_cache.clear()
    pipe._tts_audio_cache_order.clear()
    t0 = time.perf_counter()
    _ = pipe.synthesize(tgt, "en")
    tts_first_ms = (time.perf_counter() - t0) * 1000
    tts_subseq = []
    for _ in range(9):
        t0 = time.perf_counter()
        _ = pipe.synthesize(tgt, "en")  # cache hit on TTS audio cache
        tts_subseq.append((time.perf_counter() - t0) * 1000)
    mem["after_tts_bench"] = rss_mb()

    # T1 fingerprint extraction
    fp_ms = []
    if pipe._utt_cache is not None:
        for _ in range(100):
            t0 = time.perf_counter()
            fp = pipe._utt_cache.fingerprint(audio, sr)
            fp_ms.append((time.perf_counter() - t0) * 1000)

        # Populate cache with N=128 to test lookup-at-scale
        for i in range(128):
            jitter = audio + np.random.normal(0, 0.001, size=audio.shape).astype(np.float32)
            fpj = pipe._utt_cache.fingerprint(jitter, sr)
            pipe._utt_cache.put(fpj, "en", {"vi": f"x{i}", "tgt": f"y{i}",
                                             "audio": (16000, np.zeros(100, dtype=np.float32))})

        # Lookup-only (linear scan over N=128)
        lookup_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = pipe._utt_cache.lookup(fp, "en")
            lookup_ms.append((time.perf_counter() - t0) * 1000)
    else:
        lookup_ms = []

    # T2 fingerprint + lookup
    t2_ms = []
    if pipe._tm is not None:
        from translation_memory import text_fingerprint
        sample_text = vi or "xin chào"
        for _ in range(100):
            t0 = time.perf_counter()
            _ = text_fingerprint(sample_text)
            _ = pipe._tm.lookup(sample_text, "en")
            t2_ms.append((time.perf_counter() - t0) * 1000)

    # TTS scaling by output text length
    tts_scaling = {}
    pipe._tts_audio_cache.clear()
    pipe._tts_audio_cache_order.clear()
    for nw in (5, 10, 20, 50):
        text = " ".join(["hello"] * nw)
        t0 = time.perf_counter()
        _ = pipe.synthesize(text, "en")
        ms = (time.perf_counter() - t0) * 1000
        tts_scaling[nw] = {"total_ms": round(ms, 1), "ms_per_word": round(ms / nw, 1)}
        pipe._tts_audio_cache.clear()
        pipe._tts_audio_cache_order.clear()

    report["per_stage"] = {
        "asr_steady":      _stats(asr_ms),
        "mt_steady":       _stats(mt_ms),
        "tts_first_ms":    round(tts_first_ms, 1),
        "tts_subseq":      _stats(tts_subseq),
        "t1_fingerprint":  _stats(fp_ms),
        "t1_lookup_n128":  _stats(lookup_ms),
        "t2_fp_lookup":    _stats(t2_ms),
        "tts_scaling":     tts_scaling,
    }

    # ---- 6. CPU utilization snapshot ----
    n_cores = psutil.cpu_count(logical=True) if HAVE_PSUTIL else 1
    cpu_samples_nz = [x for x in cpu_samples if x > 0]
    cpu_max = max(cpu_samples_nz) if cpu_samples_nz else 0
    cpu_p95 = _percentile(cpu_samples_nz, 0.95) if cpu_samples_nz else 0
    cpu_mean = sum(cpu_samples_nz) / len(cpu_samples_nz) if cpu_samples_nz else 0
    report["cpu"] = {
        "logical_cores": n_cores,
        "max_pct":       round(cpu_max, 1),
        "p95_pct":       round(cpu_p95, 1),
        "mean_pct":      round(cpu_mean, 1),
        "samples":       len(cpu_samples),
        # 100% per core. Available is n_cores * 100%.
        "fraction_of_total": round(cpu_max / (n_cores * 100), 3) if cpu_max else 0,
    }
    report["memory_mb"] = mem

    # ---- 7. Save JSON ----
    out_json = RESULTS / "profile_pipeline.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # ---- 8. Markdown report with recommendations ----
    top3 = cold_top[:3]
    asr_p50 = report["per_stage"]["asr_steady"]["p50"]
    mt_p50  = report["per_stage"]["mt_steady"]["p50"]
    tts_p50_first = tts_first_ms
    fp_p50 = report["per_stage"]["t1_fingerprint"]["p50"] if fp_ms else 0
    lk_p50 = report["per_stage"]["t1_lookup_n128"]["p50"] if lookup_ms else 0

    md = []
    md.append("# Pipeline cache-miss profile")
    md.append("")
    md.append(f"Audio: `{AUDIO_PATH.name}` duration={duration:.2f}s")
    md.append(f"Cold cache-miss total: **{cold_ms:.0f} ms** "
              f"(asr={res_cold['latency']['asr_ms']:.0f} / "
              f"mt={res_cold['latency']['mt_ms']:.0f} / "
              f"tts={res_cold['latency']['tts_ms']:.0f})")
    md.append(f"Warm cache-hit total: **{warm_ms:.1f} ms**")
    md.append("")
    md.append("## Top 3 hot functions (cumulative time, cold run)")
    for i, f in enumerate(top3, 1):
        md.append(f"{i}. `{f['func']}` — cum={f['cumtime']:.3f}s "
                  f"tot={f['tottime']:.3f}s calls={f['ncalls']}")
    md.append("")
    md.append("## Per-stage steady-state (p50 / p95 ms)")
    md.append(f"- ASR transcribe: p50={asr_p50:.0f}  p95={report['per_stage']['asr_steady']['p95']:.0f}")
    md.append(f"- MT translate:   p50={mt_p50:.0f}  p95={report['per_stage']['mt_steady']['p95']:.0f}")
    md.append(f"- TTS first synth: {tts_p50_first:.0f}  (subsequent cached: "
              f"p50={report['per_stage']['tts_subseq']['p50']:.1f})")
    md.append(f"- T1 fingerprint (100x):    p50={fp_p50:.2f}  p95={report['per_stage']['t1_fingerprint']['p95']:.2f}")
    md.append(f"- T1 lookup N=128 (100x):   p50={lk_p50:.3f}  p95={report['per_stage']['t1_lookup_n128']['p95']:.3f}")
    md.append("")
    md.append("## TTS scaling by output text length")
    for nw, st in tts_scaling.items():
        md.append(f"- {nw} words → {st['total_ms']:.0f} ms ({st['ms_per_word']:.1f} ms/word)")
    md.append("")
    md.append("## CPU utilization (cold run)")
    md.append(f"- Logical cores: {n_cores}")
    md.append(f"- Max sampled CPU%: {cpu_max:.0f} ({100*cpu_max/(n_cores*100):.0f}% of total available)")
    md.append(f"- p95 CPU%: {cpu_p95:.0f}")
    md.append(f"- Mean CPU%: {cpu_mean:.0f}")
    md.append("")
    md.append("## Memory (RSS MB)")
    for k, v in mem.items():
        md.append(f"- {k}: {v} MB")
    md.append("")
    md.append("## Recommendations")

    recs = []
    if asr_p50 > 3000:
        recs.append(f"ASR p50={asr_p50:.0f}ms is the dominant remaining stage. "
                    "Options: (a) switch PhoWhisper-small INT8 → PhoWhisper-tiny "
                    "(if WER acceptable), (b) raise CT2 intra_threads (currently 0=auto, "
                    "test 4/8/12), (c) chunk audio + emit segments to overlap with MT, "
                    "(d) reduce no_repeat_ngram_size + repetition_penalty cost.")
    if mt_p50 > 2000:
        recs.append(f"MT p50={mt_p50:.0f}ms still high. Lower max_decoding_length cap or "
                    "use 8-bit quantization with INT8_FLOAT16 (faster on AVX2).")
    if tts_p50_first > 3000:
        recs.append(f"TTS first-synth p50={tts_p50_first:.0f}ms — pre-tokenize the 42 "
                    "industrial phrases at startup, warm tokenizer; the cold call includes "
                    "tokenizer + VITS forward.")
    if fp_p50 > 50:
        recs.append(f"T1 fingerprint p50={fp_p50:.1f}ms — non-trivial overhead on every "
                    "miss. Most cost is in _f0_contour autocorrelation loop. Vectorize "
                    "the per-frame loop using stride_tricks.sliding_window_view.")
    if lk_p50 > 5:
        recs.append(f"T1 lookup at N=128 takes p50={lk_p50:.2f}ms — linear scan. "
                    "Stack fingerprints into a (N,272) matrix and do batched dot-product "
                    "(numpy matmul) for O(1) Python overhead.")
    if HAVE_PSUTIL and cpu_max < 0.5 * n_cores * 100:
        recs.append(f"CPU underused (max {cpu_max:.0f}% of {n_cores*100}% available). "
                    "ASR and MT both run sequentially on single CT2 instance; consider "
                    "launching TTS bg-preload, and bump VOICEAI_CT2_INTRA_THREADS to "
                    f"{n_cores//2} or {n_cores}.")
    # General quick wins
    recs.append("Pre-tokenize industrial phrases at startup (skip tokenizer call on "
                "warm TTS path) — see industrial_phrases.json.")
    recs.append("Disable Whisper output_hidden_states (faster-whisper does this already, "
                "but verify ASR config does not enable diagnostic outputs).")

    for r in recs:
        md.append(f"- {r}")

    md_path = RESULTS / "profile_pipeline_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
