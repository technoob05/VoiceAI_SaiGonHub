"""Edge audio front-end — learned VAD + speech denoiser, both ONNX/on-device.

Two capabilities, one download pattern, all via sherpa-onnx (Apache-2.0), no
PyTorch:

  * SileroVAD — a LEARNED speech/non-speech discriminator used as a confirmation
    stage AFTER our cheap tone-energy VAD. Our tone-VAD gates on harmonic energy
    and will pass broadband factory noise; Silero rejects it. (Two-stage cascade
    idea from KoljaB/RealtimeSTT, MIT; model + runtime from sherpa-onnx.)
  * GtcrnDenoiser — GTCRN speech enhancement as an industrial-noise front-end
    placed BEFORE ASR. Tiny grouped-conv-recurrent net, CPU real-time, 16 kHz.
    (sherpa-onnx OfflineSpeechDenoiser; example offline-speech-enhancement-gtcrn.py)

Both run under onnxruntime — the same runtime as our Piper TTS — which is the
first step of collapsing the whole pipeline onto one runtime that later ports to
the Qualcomm Hexagon QNN backend.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
# Overridable because this module is also imported from read-only mounts (a Kaggle
# dataset input) and from the device supervisor, where the weights do not sit
# under the source tree and there is no network to fall back on.
MODELS_ROOT = Path(os.environ.get("VOICEAI_FRONTEND_MODELS_DIR", "").strip()
                   or PROJECT / "data" / "raw" / "onnx-frontend")

SILERO_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
GTCRN_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speech-enhancement-models/gtcrn_simple.onnx")


def _ensure(url: str, name: Optional[str] = None) -> str:
    dest = MODELS_ROOT / (name or url.rsplit("/", 1)[-1])
    if dest.exists() and dest.stat().st_size >= 4096:
        return str(dest)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size < 4096:
        print(f"[edge_frontend] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return str(dest)


def _to_16k_mono(audio: np.ndarray, sr: int) -> "tuple[np.ndarray, int]":
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if sr != 16_000:
        n = int(round(len(audio) * 16_000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16_000
    return np.ascontiguousarray(audio), sr


class SileroVAD:
    """Learned VAD confirmation stage (sherpa-onnx VoiceActivityDetector)."""

    def __init__(self, min_silence: float = 0.25, threshold: float = 0.5,
                 num_threads: int = 1):
        import sherpa_onnx
        model = _ensure(SILERO_URL)
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.threshold = threshold
        cfg.sample_rate = 16_000
        cfg.num_threads = num_threads
        self._cfg = cfg
        self._window = cfg.silero_vad.window_size
        self._sherpa = sherpa_onnx

    def _detector(self):
        return self._sherpa.VoiceActivityDetector(self._cfg, buffer_size_in_seconds=100)

    def segments(self, audio: np.ndarray, sr: int) -> "list[tuple[float, float]]":
        audio, sr = _to_16k_mono(audio, sr)
        vad = self._detector()
        out: list = []
        i = 0
        while i + self._window <= len(audio):
            vad.accept_waveform(audio[i:i + self._window])
            i += self._window
            while not vad.empty():
                seg = vad.front
                out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
                vad.pop()
        vad.flush()
        while not vad.empty():
            seg = vad.front
            out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
            vad.pop()
        return out

    def is_speech(self, audio: np.ndarray, sr: int, min_speech: float = 0.15) -> bool:
        """True if any confirmed speech segment ≥ min_speech seconds is present."""
        return any((e - s) >= min_speech for s, e in self.segments(audio, sr))


class GtcrnDenoiser:
    """GTCRN speech enhancement front-end (sherpa-onnx OfflineSpeechDenoiser)."""

    def __init__(self, num_threads: int = 1):
        import sherpa_onnx
        model = _ensure(GTCRN_URL)
        cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model),
                num_threads=num_threads, provider="cpu",
            )
        )
        if not cfg.validate():
            raise RuntimeError("Invalid GTCRN denoiser config")
        self._sd = sherpa_onnx.OfflineSpeechDenoiser(cfg)

    def denoise(self, audio: np.ndarray, sr: int) -> "tuple[np.ndarray, int]":
        if audio.ndim > 1:
            audio = audio[:, 0]
        out = self._sd(np.ascontiguousarray(audio.astype(np.float32)), sr)
        return np.asarray(out.samples, dtype=np.float32), int(out.sample_rate)


class StreamingGtcrnDenoiser:
    """GTCRN as a frame-by-frame filter for the capture path.

    The whole-signal `GtcrnDenoiser` cannot sit in front of a live VAD: it needs
    the complete utterance, which is the thing the VAD is supposed to decide the
    boundaries of. This wraps sherpa-onnx's OnlineSpeechDenoiser instead, which
    consumes `frame_shift_in_samples` at a time and emits enhanced audio as it goes.

    This is the ONE place GTCRN was measured to earn its keep. On a Kaggle T4,
    Silero false-triggered on impulsive metallic noise at 8.3-12.5 % across four
    SNRs; after GTCRN, 0 % in every one (Results/noise_robustness_README.md). Each
    false trigger costs a whole wasted ASR pass. The same run found the denoiser
    is neutral-to-harmful in front of ASR, so the recogniser must keep receiving
    the RAW audio - see LiveConfig.denoise.
    """

    def __init__(self, num_threads: int = 1):
        import sherpa_onnx
        model = _ensure(GTCRN_URL)
        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model),
                num_threads=num_threads, provider="cpu",
            )
        )
        if not cfg.validate():
            raise RuntimeError("Invalid streaming GTCRN denoiser config")
        self._sd = sherpa_onnx.OnlineSpeechDenoiser(cfg)
        self.frame_shift = int(self._sd.frame_shift_in_samples)
        self.sample_rate = int(self._sd.sample_rate)
        self._pending = np.empty(0, dtype=np.float32)

    def accept(self, frame: np.ndarray, sr: int) -> np.ndarray:
        """Feed raw audio, get whatever enhanced audio is ready.

        The model consumes fixed-size frames, so buffer the remainder rather than
        padding it: padding every call would inject silence into the middle of the
        stream and the enhancement would fight it.
        """
        if frame.ndim > 1:
            frame = frame[:, 0]
        self._pending = np.concatenate(
            [self._pending, np.ascontiguousarray(frame, dtype=np.float32)])
        out = []
        while len(self._pending) >= self.frame_shift:
            chunk, self._pending = (self._pending[:self.frame_shift],
                                    self._pending[self.frame_shift:])
            out.append(np.asarray(self._sd(chunk, sr).samples, dtype=np.float32))
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Drain the model's internal buffer at the end of a stream."""
        tail = np.asarray(self._sd.flush().samples, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        return tail

    def reset(self) -> None:
        self.flush()


# Lazy singletons (models load once; ~1 MB VAD, ~1 MB GTCRN).
_VAD: Optional[SileroVAD] = None
_DENOISE: Optional[GtcrnDenoiser] = None


def get_vad() -> SileroVAD:
    global _VAD
    if _VAD is None:
        _VAD = SileroVAD()
    return _VAD


def get_denoiser() -> GtcrnDenoiser:
    global _DENOISE
    if _DENOISE is None:
        _DENOISE = GtcrnDenoiser()
    return _DENOISE


def make_streaming_denoiser(num_threads: int = 1) -> "StreamingGtcrnDenoiser":
    """Not a singleton: each capture stream owns recurrent state of its own."""
    return StreamingGtcrnDenoiser(num_threads=num_threads)


if __name__ == "__main__":
    import argparse, sys, time
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    import soundfile as sf

    ap = argparse.ArgumentParser(description="Edge audio front-end (VAD + denoiser)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vad"); v.add_argument("wav")
    d = sub.add_parser("denoise"); d.add_argument("wav"); d.add_argument("--out", default="denoised.wav")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if args.cmd == "vad":
        vad = get_vad()
        segs = vad.segments(audio, sr)
        print(f"speech = {vad.is_speech(audio, sr)}  segments={[(round(s,2),round(e,2)) for s,e in segs]}")
    else:
        den = get_denoiser()
        t0 = time.perf_counter()
        out, osr = den.denoise(audio, sr)
        rtf = (time.perf_counter() - t0) / (len(audio) / sr)
        sf.write(args.out, out, osr)
        print(f"denoised -> {args.out}  ({len(out)/osr:.2f}s @ {osr}Hz, RTF={rtf:.3f})")
