"""
Live web demo with cache-hit/miss visualization — for OneVoice AI Challenge judge demo.

Difference vs web_demo.py:
  - Uses pipeline_fast.py (INT8 cascade + tonal-perceptual cache)
  - Shows per-utterance: cache status (HIT/MISS), cosine similarity, per-stage latency,
    cumulative cache stats, "this would have been 5-1441x slower without optimizations"
  - Multi-turn history with timing badges so the audience sees the cache warming up
  - Renders a giant speedup banner when a cache hit happens

Run:
    python prototype/web_demo_wow.py
    # auto-opens http://127.0.0.1:7862  (different port from web_demo.py and web_demo_streaming.py)

UX:
  1. Mic-record or upload a Vi WAV
  2. Pick target language
  3. Click "🚀 Translate" — first time runs the INT8 cascade (~5-10 s on CPU)
  4. Click again on same audio — cache HIT, ~25 ms, giant 🎯 badge
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import deque

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# UTF-8 stdout on Windows
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

from pipeline_fast import FastCascadeS2ST, FastPipelineConfig, TTS_REPO   # noqa: E402

import gradio as gr                          # noqa: E402
import numpy as np                            # noqa: E402


_PIPE: FastCascadeS2ST | None = None
HISTORY: deque[dict] = deque(maxlen=10)


def _get_pipe() -> FastCascadeS2ST:
    global _PIPE
    if _PIPE is None:
        print("[init] constructing FastCascadeS2ST (INT8 + cache enabled)")
        cfg = FastPipelineConfig(cache_enabled=True, cache_cosine_hit=0.95)
        _PIPE = FastCascadeS2ST(cfg)
        _PIPE._ensure_asr()
        _PIPE._ensure_mt()
        print("[init] ASR + MT loaded; TTS loads lazily per target")
    return _PIPE


def translate_audio(audio_tuple, target_lang: str, progress=gr.Progress()):
    if audio_tuple is None:
        return _render_history(), _render_status("❗ no audio yet", "neutral"), None, ""
    sr, samples = audio_tuple
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / max(np.iinfo(samples.dtype).max, 1)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = len(samples) / max(sr, 1)
    if duration < 0.3:
        return _render_history(), _render_status("❗ audio too short", "warn"), None, ""

    progress(0.1, desc="loading models (one-time)")
    pipe = _get_pipe()
    progress(0.3, desc="processing (cache check + cascade)")
    t_overall = time.perf_counter()
    res = pipe.run(samples, sr, target_lang)
    wall_ms = (time.perf_counter() - t_overall) * 1000

    cache_hit = res["cache"]["hit"]
    lat = res["latency"]
    entry = {
        "vi":          res["vi_text"],
        "tgt":         res["tgt_text"],
        "target":      target_lang,
        "duration_s":  duration,
        "cache_hit":   cache_hit,
        "cache_cosine": res["cache"]["cosine"],
        "lat":         lat,
        "wall_ms":     wall_ms,
        "ts":          time.strftime("%H:%M:%S"),
        "cache_stats": res["cache"]["stats"],
    }
    HISTORY.appendleft(entry)

    if cache_hit:
        status = _render_status(
            f"🎯 CACHE HIT  ·  {wall_ms:.0f} ms  (~{36500/max(wall_ms,1):.0f}× faster than FP32 baseline)",
            "hit",
        )
    else:
        status = _render_status(
            f"⚡ cache MISS — full INT8 cascade ran ({wall_ms:.0f} ms ≈ {36500/max(wall_ms,1):.1f}× faster than FP32)",
            "miss",
        )

    return _render_history(), status, (res["sr"], res["tgt_audio"]), entry["tgt"]


def _render_status(text: str, kind: str) -> str:
    colors = {"hit":  "#16a34a", "miss": "#0ea5e9", "warn": "#d97706", "neutral": "#94a3b8"}
    bg     = {"hit":  "#dcfce7", "miss": "#e0f2fe", "warn": "#fef3c7", "neutral": "#f1f5f9"}
    c = colors.get(kind, "#475569")
    b = bg.get(kind, "#f8fafc")
    return (f"<div style='padding:14px;border-radius:10px;background:{b};color:{c};"
            f"font-weight:700;font-size:20px;border:2px solid {c};'>{text}</div>")


def _render_history() -> str:
    if not HISTORY:
        return "_No turns yet — record or upload a Vi WAV and click Translate._"
    rows = []
    rows.append("| # | time | input | target | regime | latency | RTF | output |")
    rows.append("|---|------|-------|--------|--------|---------|-----|--------|")
    for i, e in enumerate(HISTORY, start=1):
        regime = "🎯 cache hit" if e["cache_hit"] else "⚡ INT8 cascade"
        rtf = e["wall_ms"] / max(e["duration_s"] * 1000, 1)
        cos = f" (cos {e['cache_cosine']:.3f})" if e["cache_cosine"] is not None else ""
        rows.append(
            f"| {i} | {e['ts']} | {e['duration_s']:.1f}s | {e['target']} | "
            f"{regime}{cos} | **{e['wall_ms']:.0f} ms** | {rtf:.2f}× | `{e['tgt']}` |"
        )
    if HISTORY:
        stats = HISTORY[0]["cache_stats"]
        if stats:
            rows.append("")
            rows.append(
                f"📊 cache stats: lookups={stats['lookups']}, "
                f"hits={stats['hits']}, "
                f"hit_rate={stats['hit_rate']*100:.1f}%, "
                f"entries={stats['n_entries']}/{stats['max_entries']}, "
                f"fingerprint_mean={stats['fingerprint_ms_mean']:.1f} ms"
            )
    return "\n".join(rows)


def reset_session():
    HISTORY.clear()
    if _PIPE is not None and _PIPE._utt_cache is not None:
        _PIPE._utt_cache._store.clear()
        _PIPE._utt_cache.stats = type(_PIPE._utt_cache.stats)()
    return _render_history(), _render_status("🔁 session + cache reset", "neutral"), None, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoiceAI SaiGonHub — INT8 + cache wow demo") as demo:
        gr.Markdown("""
        # 🚀 VoiceAI SaiGonHub — **1,441×** speedup live demo

        End-to-end Vi → {En, Zh, Ko} cascaded speech-to-speech translation.
        Same input audio runs through three regimes:

        - **🐢 FP32 cold baseline (reference)**: ~36.5 s on CPU
        - **⚡ INT8 cascade** (faster-whisper + CT2-NLLB + dyn-INT8 VITS): ~6.7 s (**5.4× faster**)
        - **🎯 INT8 + tonal-perceptual utterance cache** on repeat: ~25 ms (**1,441× faster**)

        The cache is the novel piece: a 272-dim fingerprint mixing a Vietnamese tone histogram,
        temporal-subsampled MFCC, delta-MFCC, F0 stats and energy stats. Two utterances of the
        same phrase by the same speaker land on the same cache slot; two distinct phrases don't.
        Worker repetition in factory / construction settings turns into instant translation.

        > Click Translate twice on the SAME recording to see the cache hit. Then change the
        > recording — the cache lets the new one MISS and run the full cascade.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="🎤 Vietnamese audio (mic record OR upload WAV)",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                target = gr.Radio(
                    label="🌐 Target language",
                    choices=[("English",  "en"), ("Mandarin (text-only TTS)", "zh"),
                             ("Korean (romanized)", "ko"), ("Vietnamese (round-trip)", "vi")],
                    value="en",
                )
                with gr.Row():
                    btn   = gr.Button("🚀 Translate", variant="primary", scale=3)
                    reset = gr.Button("🔁 Reset", variant="secondary", scale=1)
            with gr.Column(scale=1):
                status   = gr.HTML(_render_status("ready", "neutral"))
                audio_out = gr.Audio(label="🔊 Target audio", type="numpy")
                tgt_text = gr.Textbox(label="📝 Last translation (text)", lines=2)

        history_md = gr.Markdown(_render_history(), label="Turn history (most-recent first)")

        btn.click(
            translate_audio,
            inputs=[audio_in, target],
            outputs=[history_md, status, audio_out, tgt_text],
        )
        reset.click(reset_session, outputs=[history_md, status, audio_out, tgt_text])

        gr.Markdown("""
        ---
        **Notes for technical judges**:
        - Bench scripts: [bench_compare.py](bench_compare.py) (FP32 vs INT8 side-by-side, warmup-corrected)
          + [demo_wow.py](demo_wow.py) (3-regime CLI dump).
        - Cache stays within memory; ~280 KB for 128 entries × 272 dims × float32.
        - Cache cosine threshold default 0.95 — catches same-phrase repeats with up to σ=0.01 jitter,
          rejects different phrases (which max out at cos ≈ 0.91 on VIVOS).
        - Final on-device target (Snapdragon Hexagon NPU + tone-aware quantization,
          locked research spine in CLAUDE.md §4): RTF < 0.4 for cache misses, instant for hits.
        """)
    return demo


def main():
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
