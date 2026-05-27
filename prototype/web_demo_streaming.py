"""
Streaming Gradio web demo with silero-VAD endpoint detection.

Difference vs web_demo.py:
  - Continuous mic input via gr.Audio(streaming=True) — chunks fire as user speaks.
  - silero-VAD segments the stream into utterances: starts when speech detected,
    finalizes after MIN_SILENCE_MS of silence.
  - Each finalized utterance auto-triggers the cascade ASR -> MT -> TTS.
  - Conversational log shows the rolling history of (vi_text, target_text) turns.

Run:
    pip install silero-vad
    python prototype/web_demo_streaming.py

UX:
  1. Click "Start streaming" → mic begins capturing
  2. Speak Vietnamese
  3. Stop talking → after ~800 ms of silence, utterance auto-finalizes
  4. Pipeline runs in background; output appears in the history below

Latency note: each utterance still costs ~25-100 s on CPU fp32 — VAD just frees
the user from clicking buttons. Real-time feel comes after we deploy the
tone-aware quantized models on Hexagon NPU.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import deque

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import CascadeS2ST, PipelineConfig, TTS_REPO   # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import gradio as gr                                          # noqa: E402
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

VAD_SR = 16000                  # silero-VAD operates at 16 kHz
CHUNK_MS = 30                    # 30 ms frames (standard VAD frame size)
MIN_SPEECH_MS = 300              # require this much speech before considering an utterance "real"
MIN_SILENCE_MS = 800             # finalize utterance after this long a pause
MAX_UTTERANCE_S = 20             # safety: force-finalize if user keeps talking


_PIPE: CascadeS2ST | None = None
_VAD = None
_VAD_GET_TIMESTAMPS = None


def _get_pipe() -> CascadeS2ST:
    global _PIPE
    if _PIPE is None:
        print("[init] constructing CascadeS2ST (CPU fp32 by default)")
        _PIPE = CascadeS2ST(PipelineConfig())
        _PIPE._ensure_asr()
        _PIPE._ensure_mt()
        print("[init] ASR + MT loaded; TTS loads lazily per target language")
    return _PIPE


def _get_vad():
    global _VAD, _VAD_GET_TIMESTAMPS
    if _VAD is None:
        print("[init] loading silero-VAD")
        from silero_vad import load_silero_vad, get_speech_timestamps
        _VAD = load_silero_vad(onnx=True)
        _VAD_GET_TIMESTAMPS = get_speech_timestamps
        print("[init] silero-VAD ready")
    return _VAD, _VAD_GET_TIMESTAMPS


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32, copy=False)
    target_len = int(round(len(audio) * 16000 / sr))
    x = np.linspace(0, len(audio), target_len, endpoint=False)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


# ---------------- Session state ----------------
class SessionBuffer:
    """Accumulates audio across streaming chunks; runs VAD on the rolling window."""

    def __init__(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.last_speech_t: float | None = None
        self.utterance_start_t: float | None = None

    def add(self, audio: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, audio])

    def reset(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.last_speech_t = None
        self.utterance_start_t = None

    def duration_s(self) -> float:
        return len(self.audio) / VAD_SR


def _ms_to_samples(ms: int) -> int:
    return int(ms * VAD_SR / 1000)


def _detect_utterance_boundary(buf: SessionBuffer) -> tuple[bool, np.ndarray | None]:
    """Run VAD over `buf.audio`; if last detected speech segment ended >MIN_SILENCE_MS ago,
    return (True, utterance_audio). Otherwise (False, None).
    """
    if len(buf.audio) < _ms_to_samples(MIN_SPEECH_MS):
        return False, None
    if buf.duration_s() > MAX_UTTERANCE_S:
        # Force-finalize at hard cap so we don't blow memory / latency budget.
        utt = buf.audio.copy()
        buf.reset()
        return True, utt

    model, get_timestamps = _get_vad()
    audio_t = torch.from_numpy(buf.audio)
    ts = get_timestamps(audio_t, model, sampling_rate=VAD_SR,
                       min_silence_duration_ms=MIN_SILENCE_MS,
                       min_speech_duration_ms=MIN_SPEECH_MS,
                       return_seconds=False)
    if not ts:
        return False, None

    last_end = ts[-1]["end"]
    silence_since_speech = len(buf.audio) - last_end
    if silence_since_speech >= _ms_to_samples(MIN_SILENCE_MS):
        first_start = ts[0]["start"]
        utt = buf.audio[first_start:last_end].copy()
        buf.reset()
        return True, utt

    return False, None


def on_stream(audio_chunk, buf_state, history, target_lang, progress=gr.Progress()):
    """Streaming-chunk callback. Returns updated state + UI panels."""
    history = history or []

    if audio_chunk is None or buf_state is None:
        return buf_state, history, history_render(history), "🎙️ Waiting for audio..."

    sr, samples = audio_chunk
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / max(np.iinfo(samples.dtype).max, 1)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = _resample_to_16k(samples, sr)

    buf_state.add(samples)
    finalized, utt = _detect_utterance_boundary(buf_state)

    if not finalized:
        return buf_state, history, history_render(history), (
            f"🎤 listening... buffer={buf_state.duration_s():.1f}s"
        )

    dur = len(utt) / VAD_SR
    progress(0.10, desc=f"VAD finalized utterance ({dur:.2f}s)")
    pipe = _get_pipe()
    progress(0.30, desc="ASR (PhoWhisper)")
    t0 = time.perf_counter()
    vi_text = pipe.transcribe(utt, VAD_SR)
    t1 = time.perf_counter()
    if not vi_text.strip():
        return buf_state, history, history_render(history), "🤐 no speech detected, listening..."

    progress(0.60, desc="MT (NLLB)")
    tgt_text = pipe.translate(vi_text, target_lang, source="vi")
    t2 = time.perf_counter()
    progress(0.85, desc=f"TTS ({target_lang})")
    tgt_audio, out_sr = pipe.synthesize(tgt_text, target_lang)
    t3 = time.perf_counter()

    turn = {
        "vi":      vi_text,
        "tgt":     tgt_text,
        "audio":   (out_sr, tgt_audio),
        "lang":    target_lang,
        "dur":     dur,
        "lat":     {"asr": (t1-t0)*1000, "mt": (t2-t1)*1000, "tts": (t3-t2)*1000, "total": (t3-t0)*1000},
    }
    history.append(turn)
    status = f"✅ utterance {len(history)} translated ({turn['lat']['total']/1000:.1f}s for {dur:.1f}s of speech)"
    return buf_state, history, history_render(history), status


def history_render(history: list[dict]) -> str:
    if not history:
        return "_Speak something — your translations will appear here._"
    lines = []
    for i, t in enumerate(history[-10:], start=1):
        lines.append(
            f"**Turn {i}** &nbsp;·&nbsp; *{t['dur']:.1f}s of Vi → {t['lang']}*&nbsp; "
            f"`asr={t['lat']['asr']:.0f}ms mt={t['lat']['mt']:.0f}ms tts={t['lat']['tts']:.0f}ms`"
        )
        lines.append(f"&nbsp;&nbsp;🇻🇳 `{t['vi']}`")
        lines.append(f"&nbsp;&nbsp;🌐 **{t['tgt']}**")
        lines.append("")
    return "\n".join(lines)


def reset_session(buf_state):
    buf = SessionBuffer()
    return buf, [], history_render([]), "🔁 Session reset. Click Record to start streaming."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoiceAI SaiGonHub — Streaming S2ST") as demo:
        gr.Markdown("""
        # 🎙️ VoiceAI SaiGonHub — **Streaming** Vi → {En, Zh, Ko} demo

        Continuous-listening cascade with **silero-VAD** endpoint detection.
        Speak; once you pause **~800 ms**, the cascade fires automatically and the
        translation appears in the rolling history below. No buttons to click between turns.

        > Each turn still costs ~25–100 s on CPU fp32 — VAD only removes the human-in-the-loop click,
        > not the model latency. That comes after the tone-aware quantization spine ships to Hexagon NPU.
        """)
        buf_state = gr.State(SessionBuffer())
        history   = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                mic = gr.Audio(
                    label="🎤 Streaming mic (continuous)",
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                )
                target = gr.Radio(
                    label="Target language",
                    choices=[("English", "en"),
                             ("Mandarin (text only)", "zh"),
                             ("Korean", "ko"),
                             ("Vietnamese (round-trip)", "vi")],
                    value="en",
                )
                reset_btn = gr.Button("🔁 Reset session", variant="secondary")
                status = gr.Markdown("🎙️ Click record on the mic to start streaming")
            with gr.Column(scale=1):
                hist_md = gr.Markdown(history_render([]))

        mic.stream(
            on_stream,
            inputs=[mic, buf_state, history, target],
            outputs=[buf_state, history, hist_md, status],
            stream_every=0.5,    # process buffer every 500 ms
        )

        reset_btn.click(
            reset_session,
            inputs=[buf_state],
            outputs=[buf_state, history, hist_md, status],
        )

        gr.Markdown("""
        ---
        **VAD config** (in code): MIN_SPEECH_MS=300, MIN_SILENCE_MS=800, MAX_UTTERANCE_S=20.

        **Notes**:
        - Mandarin currently has no MMS-TTS model → text-only output. We'll wire MeloTTS / ChatTTS in a later iteration.
        - Korean MMS-TTS expects romanized Latin — pipeline.py runs `unidecode` on Hangul output.
        - On Snapdragon Hexagon NPU + tone-aware quantization (the research contribution), per-turn latency targets <1.5 s.
        """)
    return demo


def main():
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7861,         # different port from web_demo.py (:7860)
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
