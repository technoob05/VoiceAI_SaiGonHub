"""
Gradio web demo for the VoiceAI_SaiGonHub cascaded S2ST prototype.

Launches a browser-based UI:
  - Record from microphone OR upload a Vietnamese WAV
  - Pick target language (en / zh / ko)
  - See the Vi transcript, target-language translation, and target-language audio
  - See per-stage latency

Run:
    python prototype/web_demo.py
    # browser opens at http://127.0.0.1:7860

First request triggers model downloads (~3.5 GB into data/raw/hf_cache/).
Subsequent requests are offline + faster (still ~100 s on CPU fp32; this is the
fp32 baseline — the tone-aware quantization spine will compress this to ~1 s on
Hexagon NPU later).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Reuse the env-setup + pipeline class from pipeline.py — same module, same caches.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import CascadeS2ST, PipelineConfig, TTS_REPO   # noqa: E402

# Force stdout to UTF-8 so we can print Vietnamese on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try: _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import gradio as gr                          # noqa: E402
import numpy as np                            # noqa: E402

# Singleton pipeline. First request loads ASR + MT. TTS lazy-loads per target.
_PIPE: CascadeS2ST | None = None


def _get_pipe() -> CascadeS2ST:
    global _PIPE
    if _PIPE is None:
        print("[init] constructing CascadeS2ST (CPU fp32 by default)")
        _PIPE = CascadeS2ST(PipelineConfig())
        _PIPE._ensure_asr()    # eager load — keeps first user-visible request snappy
        _PIPE._ensure_mt()
        print("[init] ASR + MT loaded; TTS loads lazily per target language")
    return _PIPE


def translate_audio(audio_tuple, target_lang: str, progress=gr.Progress()):
    """Gradio callback. `audio_tuple` is (sample_rate:int, samples:np.int16-or-float32)."""
    if audio_tuple is None:
        return "", "", None, "❗ No audio. Record or upload a WAV first."

    sr, samples = audio_tuple
    # Gradio gives int16 for mic, float32 for upload. Normalize to float32 in [-1, 1].
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = len(samples) / max(sr, 1)
    if duration < 0.3:
        return "", "", None, "❗ Audio too short (<0.3s). Speak a sentence."

    progress(0.05, desc="loading models (first call is slow)")
    pipe = _get_pipe()

    progress(0.20, desc="ASR (PhoWhisper)")
    t0 = time.perf_counter()
    vi_text = pipe.transcribe(samples, sr)
    t1 = time.perf_counter()

    progress(0.55, desc="MT (NLLB)")
    tgt_text = pipe.translate(vi_text, target_lang, source="vi")
    t2 = time.perf_counter()

    progress(0.80, desc=f"TTS ({TTS_REPO[target_lang]})")
    tgt_audio, out_sr = pipe.synthesize(tgt_text, target_lang)
    t3 = time.perf_counter()

    total_in_ms  = duration * 1000
    total_out_ms = (t3 - t0) * 1000
    rtf = total_out_ms / max(total_in_ms, 1)
    summary = (
        f"input duration  = {duration:.2f} s @ {sr} Hz\n"
        f"asr   = {(t1-t0)*1000:>7.0f} ms   (PhoWhisper-small)\n"
        f"mt    = {(t2-t1)*1000:>7.0f} ms   (NLLB-200-distilled-600M)\n"
        f"tts   = {(t3-t2)*1000:>7.0f} ms   (MMS-TTS-{target_lang})\n"
        f"total = {total_out_ms:>7.0f} ms\n"
        f"\nRTF = {rtf:.2f}x   (>1 means slower than real-time on this device)\n"
        f"Target on Hexagon NPU after tone-aware quantization: <1.5 s (RTF ≈ 0.4)"
    )
    return vi_text, tgt_text, (out_sr, tgt_audio), summary


# ----------------- Example VIVOS samples (preloaded) -----------------
VIVOS_TEST = Path(__file__).resolve().parents[1] / "data" / "processed" / "vivos" / "vivos" / "test" / "waves"

def _find_vivos_examples(n: int = 4) -> list[list]:
    """Pick a few VIVOS test WAVs as examples (paired with default target=en)."""
    if not VIVOS_TEST.exists():
        return []
    wavs = []
    for spk in sorted(VIVOS_TEST.iterdir())[:n]:
        ws = sorted(spk.glob("*.wav"))
        if ws:
            wavs.append([str(ws[0]), "en"])
    return wavs


# ----------------- UI -----------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoiceAI SaiGonHub — Vi↔{En,Zh,Ko} S2ST prototype") as demo:
        gr.Markdown("""
        # 🎙️ VoiceAI SaiGonHub — Cascaded Speech-to-Speech Translation prototype

        Speak Vietnamese into the mic (or upload a WAV); get back text + audio in **English / Mandarin / Korean**.

        Stack: **PhoWhisper-small** (ASR) → **NLLB-200-distilled-600M** (MT) → **MMS-TTS** (VITS).
        Runs on CPU fp32 — ~100 s for a 4 s utterance (slow but correct). Target after tone-aware
        quantization on Snapdragon Hexagon NPU: <1.5 s.

        > First click triggers model load (~30–60 s). After that responses are stable.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="🎤 Vietnamese audio (mic record OR upload WAV)",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                target = gr.Radio(
                    label="Target language",
                    choices=[("English",  "en"),
                             ("Mandarin", "zh"),
                             ("Korean",   "ko"),
                             ("Vietnamese (round-trip)", "vi")],
                    value="en",
                )
                btn = gr.Button("🔁 Translate", variant="primary")
            with gr.Column(scale=1):
                vi_box  = gr.Textbox(label="🇻🇳 Vietnamese transcript (ASR output)", lines=2)
                tgt_box = gr.Textbox(label="🌐 Target translation (MT output)",     lines=2)
                audio_out = gr.Audio(label="🔊 Synthesized target audio (TTS output)", type="numpy")
                lat_box = gr.Textbox(label="⏱️ Latency breakdown", lines=8)

        examples = _find_vivos_examples(4)
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[audio_in, target],
                label="Examples from VIVOS test set",
            )

        btn.click(
            translate_audio,
            inputs=[audio_in, target],
            outputs=[vi_box, tgt_box, audio_out, lat_box],
        )

        gr.Markdown("""
        ---
        **Caveats** (this is the fp32 baseline, not the final method):
        - Mic input is **browser-native sample rate (typically 44.1 kHz or 48 kHz)** → resampled to 16 kHz inside the cascade. Long-form recordings (>30 s) get clipped by PhoWhisper's input limit.
        - No streaming yet (one-shot per click). Streaming chunked decode is on the post-quantization roadmap.
        - The TTS voice for Mandarin / Korean / Vietnamese targets is a generic MMS-TTS voice, **not** the original speaker's voice. Voice-cloning is out of scope for the prototype.
        - **Tone preservation under quantization is the research contribution** — this fp32 baseline is what we'll quantize against in `exp/` scripts.
        """)
    return demo


def main():
    demo = build_ui()
    # `queue()` serializes requests so we never run two cascades concurrently
    # (CPU fp32 OOMs easily under contention).
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
