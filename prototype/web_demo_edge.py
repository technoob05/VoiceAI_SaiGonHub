"""
Polished Gradio web demo for VoiceAI SaiGonHub — Edge speech-to-speech translation.

100% on-device cascade (denoise → VAD → LocalAgreement streaming ASR → NLLB MT →
streaming TTS), driven by prototype/pipeline_fast.py::FastCascadeS2ST.

Difference vs web_demo_streaming.py:
  - Uses the FAST INT8 pipeline (pipeline_fast.FastCascadeS2ST), not the slow FP32 one.
  - Single-shot upload/mic → click Translate → live streamed UI updates via a Python
    generator (Vi transcript, target translation, synthesized target speech, metrics).
  - Runtime toggles (denoise / Silero-VAD confirm / streaming-TTS / TTS backend) rebuild
    a warm-cached pipeline per config combo (flags live on the dataclass config).

Run:
    python prototype/web_demo_edge.py
    # then open http://127.0.0.1:7862
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Force UTF-8 stdout/stderr (Windows cp1252) so Vietnamese text prints cleanly.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np                                              # noqa: E402


def _ensure_ffmpeg() -> None:
    """Gradio decodes browser mic recordings (webm/opus) with the `ffmpeg` command.
    Without ffmpeg on PATH, MICROPHONE recording silently fails (upload of a .wav still
    works — no decode needed). We ship one via the `imageio-ffmpeg` wheel and expose it
    as `ffmpeg`/`ffmpeg.exe` on PATH so the subprocess call finds it."""
    import shutil
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        print(f"[ui] ffmpeg not available ({e}); mic recording may not work — pip install imageio-ffmpeg")
        return
    dst_dir = Path(os.environ.get("TEMP", ".")) / "voiceai_ffmpeg"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        if not dst.exists():
            shutil.copy2(src, dst)
        os.environ["PATH"] = str(dst_dir) + os.pathsep + os.environ.get("PATH", "")
        # pydub (used by gradio) also honors these:
        os.environ.setdefault("FFMPEG_BINARY", str(dst))
        print(f"[ui] ffmpeg ready → {dst}")
    except Exception as e:  # noqa: BLE001
        print(f"[ui] could not stage ffmpeg ({e}); mic recording may not work")


_ensure_ffmpeg()

import gradio as gr                                             # noqa: E402

from pipeline_fast import FastCascadeS2ST, FastPipelineConfig   # noqa: E402


# --------------------------------------------------------------------------------------
# UI constants
# --------------------------------------------------------------------------------------
TITLE = "VoiceAI SaiGonHub — Edge S2ST (100% on-device)"

TARGET_CHOICES = [
    ("English", "en"),
    ("Mandarin 中文", "zh"),
    ("Korean 한국어", "ko"),
    ("Vietnamese (round-trip)", "vi"),
]

SAMPLE_WAV = (
    HERE.parent / "data" / "processed" / "vivos" / "vivos"
    / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav"
)

HOW_IT_WORKS = """
### How it works — a fully offline speech-translation stack

Every stage below runs **on-device** — no audio, text, or telemetry leaves the machine.

1. **Denoise (GTCRN)** — optional learned front-end that lifts SNR on factory-floor / street
   noise before the recognizer ever sees the signal (sherpa-onnx, INT-friendly).
2. **VAD gate** — a cheap tone-energy voice-activity gate rejects silence / hum outright; an
   optional **Silero** learned-VAD confirmation stage rejects broadband noise the cheap gate
   lets through (two-stage AND).
3. **Streaming ASR (PhoWhisper-small, INT8)** — faster-whisper / CTranslate2 backend with
   LocalAgreement-style incremental decoding, emitting the Vietnamese transcript as segments
   finalize.
4. **MT (NLLB-200-distilled-600M, INT8)** — CTranslate2 translator, sentence-batched, greedy;
   translates Vi → {en, zh, ko, vi}.
5. **TTS (Piper / MMS-TTS VITS)** — optional **streaming** synthesis (fragment-synth + faded
   concat) so the first audio plays before the whole sentence is spoken.

Two perceptual caches short-circuit repeats: a **tonal audio cache** (`cache_hit`, sub-100 ms)
and a text-level **translation memory** (`tm_full_hit`), both fully local.

> Piper is the default TTS backend — it ships all four competition languages (incl. Mandarin).
> MMS-TTS has no Mandarin model, so `zh` falls back to silence under the `mms` backend.
"""


# --------------------------------------------------------------------------------------
# Single shared, always-warm pipeline
# --------------------------------------------------------------------------------------
# The four toggles (backend / denoise / silero / streaming_tts) are read at RUN time
# inside run_streaming()/synthesize(), NOT at construction — so ONE pipeline serves every
# combo. We just set the flags on its cfg before each request. (Building a fresh pipeline
# per combo would reload ASR+MT ~130 s and drop the browser connection — the bug this
# replaces.) Requests are serialized (concurrency_limit=1) so the shared cfg is safe.
_PIPE: Optional[FastCascadeS2ST] = None


def get_pipeline() -> FastCascadeS2ST:
    global _PIPE
    if _PIPE is None:
        print("[ui] building pipeline (one-time) ...", flush=True)
        cfg = FastPipelineConfig()
        # DISABLE the perceptual caches in the demo. The tonal audio-cache and the
        # fuzzy translation-memory (cosine ≥ 0.80) exist to short-circuit REPEATED
        # industrial phrases, but on a demo where you try different sentences they
        # produce false hits — returning a stale translation of a *different*
        # utterance (the `tm_full_hit` bug). Correctness > repeat-latency here.
        cfg.cache_enabled = False
        cfg.tm_enabled = False
        _PIPE = FastCascadeS2ST(cfg)
    return _PIPE


def apply_flags(pipe: FastCascadeS2ST, backend, denoise, silero, streaming_tts, punct=False) -> None:
    pipe.cfg.tts_backend = str(backend).strip().lower()
    pipe.cfg.denoise_gtcrn = bool(denoise)
    pipe.cfg.silero_confirm = bool(silero)
    pipe.cfg.streaming_tts = bool(streaming_tts)
    pipe.cfg.punct = bool(punct)


def warmup() -> None:
    """Load ASR + MT + default Piper TTS BEFORE the server accepts requests, so the first
    translation is fast and never blocks the browser long enough to look disconnected."""
    print("[ui] warming models (ASR + MT + TTS)… first load ~30-60 s", flush=True)
    pipe = get_pipeline()
    pipe._ensure_asr()
    pipe._ensure_mt()
    try:
        pipe.cfg.tts_backend = "piper"
        pipe.synthesize("khởi động", "en")
    except Exception as e:  # noqa: BLE001
        print(f"[ui] TTS warmup skipped: {e}", flush=True)
    print("[ui] warm — ready to serve.", flush=True)


# --------------------------------------------------------------------------------------
# Audio + metrics helpers
# --------------------------------------------------------------------------------------
def _to_float32_mono(data: np.ndarray) -> np.ndarray:
    """Gradio numpy audio is often int16 (and may be stereo). Normalize to float32 mono
    in [-1, 1]. Resampling to 16 kHz happens inside run_streaming, so we keep the true sr."""
    data = np.asarray(data)
    if data.dtype.kind in ("i", "u"):
        max_mag = float(np.iinfo(data.dtype).max) or 1.0
        data = data.astype(np.float32) / max_mag
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def _as_gradio_audio(sr: int, arr: Optional[np.ndarray]):
    """Package a float32 waveform as a gradio Audio value (sr, data), or None."""
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    return (int(sr), arr)


def _render_metrics(stage: str, lat: dict, fired: list[str], cache: Optional[dict]) -> str:
    lat = lat or {}
    seen = list(dict.fromkeys(fired))  # unique, order-preserving
    path = " → ".join(f"`{s}`" for s in seen) if seen else "—"

    def ms(key):
        v = lat.get(key)
        return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

    lines = [
        f"**Stage:** `{stage}`  ·  **path:** {path}",
        "",
        "| stage | latency (ms) |",
        "|:--|--:|",
        f"| ASR | {ms('asr_ms')} |",
        f"| MT | {ms('mt_ms')} |",
        f"| TTS | {ms('tts_ms')} |",
        f"| cache probe | {ms('cache_ms')} |",
        f"| **total** | **{ms('total_ms')}** |",
    ]
    if lat.get("tts_first_ms") is not None:
        lines.insert(-1, f"| TTS first-audio | {ms('tts_first_ms')} |")

    if cache:
        if cache.get("hit"):
            lines += ["", f"⚡ **Audio-cache hit** (cosine = {cache.get('cosine')}) — instant result."]
        elif cache.get("tm_hit"):
            lines += ["", f"⚡ **Translation-memory hit** (cosine = {cache.get('tm_cosine')})."]
        if cache.get("vad_rejected"):
            conf = cache.get("vad_confidence")
            lines += ["", f"🔇 **VAD rejected** — no speech detected (confidence = {conf})."]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Core streaming generator wired to the Translate button
# --------------------------------------------------------------------------------------
def process(audio, target, backend, denoise, silero, streaming_tts, punct):
    """Generator that drives FastCascadeS2ST.run_streaming and yields progressive UI
    updates: (vi_transcript, target_translation, target_audio, metrics_markdown)."""
    if audio is None:
        yield ("", "", None, "_Please upload a file or record from the mic, then click **Translate**._")
        return

    sr, data = audio
    data = _to_float32_mono(data)
    if data.size == 0:
        yield ("", "", None, "_Empty audio — nothing to translate._")
        return

    # Immediate heartbeat BEFORE any heavy work, so the browser never sees a silent gap
    # (a long silent gap is what triggers "Connection to the server was lost").
    yield ("", "", None, f"_⏳ Running on-device pipeline ({backend}, "
                         f"denoise={'on' if denoise else 'off'})…_")

    pipe = get_pipeline()
    apply_flags(pipe, backend, denoise, silero, streaming_tts, punct)

    vi_text, tgt_text = "", ""
    fired: list[str] = []
    frag_bufs: list[np.ndarray] = []
    frag_sr = 16000
    audio_out = None

    for ev in pipe.run_streaming(data, int(sr), target):
        stage = ev.get("stage", "?")
        fired.append(stage)
        if ev.get("vi_text"):
            vi_text = ev["vi_text"]
        if ev.get("tgt_text"):
            tgt_text = ev["tgt_text"]
        lat = ev.get("latency", {})
        cache = ev.get("cache")

        if stage == "vad_reject":
            yield (vi_text, tgt_text, None, _render_metrics(stage, lat, fired, cache))
            return

        ev_audio = ev.get("tgt_audio")
        if stage == "tts_partial" and ev_audio is not None and len(ev_audio) > 0:
            # Streaming TTS: accumulate fragments and play what we have so far.
            frag_sr = int(ev.get("sr", 16000))
            frag_bufs.append(np.asarray(ev_audio, dtype=np.float32))
            audio_out = _as_gradio_audio(frag_sr, np.concatenate(frag_bufs))
        elif ev_audio is not None and len(ev_audio) > 0:
            # Complete audio (done / cache_hit / tm_full_hit).
            audio_out = _as_gradio_audio(int(ev.get("sr", 16000)), np.asarray(ev_audio, dtype=np.float32))

        yield (vi_text, tgt_text, audio_out, _render_metrics(stage, lat, fired, cache))


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(
            f"# 🎙️ {TITLE}\n"
            "Vietnamese speech → **{English, 中文, 한국어, Vietnamese}** speech, running "
            "entirely on-device. Upload or record, pick a target language, hit **Translate**, "
            "and watch the transcript → translation → synthesized speech stream in live."
        )

        with gr.Row():
            target = gr.Dropdown(
                label="Target language",
                choices=TARGET_CHOICES,
                value="en",
                scale=2,
            )
            backend = gr.Radio(
                label="TTS backend",
                choices=[("Piper (all 4 langs)", "piper"), ("MMS-TTS (no zh)", "mms")],
                value="piper",
                scale=2,
            )
            denoise = gr.Checkbox(label="Denoise (GTCRN)", value=False, scale=1)
            silero = gr.Checkbox(label="Silero VAD confirm", value=False, scale=1)
            streaming_tts = gr.Checkbox(label="Streaming TTS", value=False, scale=1)
            punct = gr.Checkbox(label="Punctuation (VN)", value=False, scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="🎤 Input (upload a file or record)",
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                translate_btn = gr.Button("Translate", variant="primary")
                if SAMPLE_WAV.exists():
                    gr.Examples(
                        examples=[[str(SAMPLE_WAV)]],
                        inputs=[audio_in],
                        label="Example (VIVOS test clip)",
                    )
            with gr.Column(scale=1):
                vi_out = gr.Textbox(
                    label="🇻🇳 Vietnamese transcript (live)",
                    lines=3,
                    interactive=False,
                )
                tgt_out = gr.Textbox(
                    label="🌐 Target translation (live)",
                    lines=3,
                    interactive=False,
                )
                audio_out = gr.Audio(
                    label="🔊 Synthesized target speech",
                    type="numpy",
                    autoplay=True,
                )
                metrics = gr.Markdown(
                    "_Latency & pipeline path will appear here after you translate._"
                )

        with gr.Accordion("How it works (on-device pipeline)", open=False):
            gr.Markdown(HOW_IT_WORKS)

        translate_btn.click(
            process,
            inputs=[audio_in, target, backend, denoise, silero, streaming_tts, punct],
            outputs=[vi_out, tgt_out, audio_out, metrics],
            concurrency_limit=1,   # shared pipeline cfg → serialize requests
        )

    return demo


def main():
    warmup()   # load models BEFORE accepting connections (no in-request 130 s stall)
    build_ui().queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        inbrowser=True,
        allowed_paths=[str(SAMPLE_WAV.parent)],   # let the VIVOS example load
    )


if __name__ == "__main__":
    main()
