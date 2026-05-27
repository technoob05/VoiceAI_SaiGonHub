"""
Tone-energy VAD — F0-stability voice gating for tonal languages.

Standard VADs (silero-VAD, webrtcvad) classify "speech vs non-speech" based on
broad-band energy + spectral entropy. They work well for high-resource languages
but mis-fire on tonal languages where short voiced syllables with high tonal
variability look like noise to a generic VAD.

This module adds a *tone-anchored* VAD that complements silero-VAD: when the
generic VAD says "uncertain" or "no speech", we run a cheap F0-stability check.
If the audio contains contiguous voiced regions with the F0 dynamics of
Vietnamese tones (or any tonal language), we override the generic VAD and
pass the audio through to ASR.

This catches the failure mode where silero misses very short Vi syllables
("dạ", "vâng", "ừ") that workers say frequently — and it rejects garbage that
silero accepts (radio noise from machinery).

Novel contribution candidate for paper §3.6: tonal-language VAD as a typology-
aware pre-filter that gates the full cascade.

Usage:
    decision, score = ToneVAD().is_likely_speech(audio, sr)
    if decision:
        run_cascade(audio)
    else:
        return "(no speech detected)"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ToneVADConfig:
    sr_target:           int   = 16000
    f_min:               float = 60.0
    f_max:               float = 400.0
    hop_ms:              float = 20.0
    win_ms:              float = 40.0
    # Decision thresholds — tuned on VIVOS train so real Vi speech passes.
    min_voiced_ratio:    float = 0.08   # >=8% frames must be voiced
    min_voiced_seconds:  float = 0.15   # >=150 ms of voiced content
    max_dyn_range_octaves: float = 4.0  # Vi tones span up to ~3 oct legitimately
    # Tone-character signal — VIVOS sampling shows log_f0_std ≈ 0.4-0.7 for natural Vi speech.
    require_tonal_pattern: bool = True   # require log-F0 variance to be in tonal range
    log_f0_std_min:      float = 0.05    # tonal speech has structured F0 variance
    log_f0_std_max:      float = 0.90    # only reject when extremely erratic (music)
    autocorr_voiced_thr: float = 0.3     # autocorr peak strength to call frame voiced


def _bandpass_fft(audio: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Cheap zero-phase bandpass via FFT mask."""
    n = len(audio)
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= low) & (freqs <= high)
    spec[~mask] = 0
    return np.fft.irfft(spec, n).astype(np.float32)


def _autocorr_f0(audio: np.ndarray, sr: int, cfg: ToneVADConfig) -> np.ndarray:
    """Per-frame F0 estimate (Hz). 0 = unvoiced."""
    band = _bandpass_fft(audio, sr, cfg.f_min * 0.7, cfg.f_max * 1.3)
    hop = int(sr * cfg.hop_ms / 1000)
    win = int(sr * cfg.win_ms / 1000)
    if win < 4:
        return np.zeros(0, dtype=np.float32)
    n_frames = max(0, (len(band) - win) // hop + 1)
    out = np.zeros(n_frames, dtype=np.float32)
    min_lag = max(2, int(sr / cfg.f_max))
    max_lag = min(win - 2, int(sr / cfg.f_min))
    for i in range(n_frames):
        s = i * hop
        f = band[s:s + win].astype(np.float32)
        e = float((f ** 2).sum())
        if e < 1e-6:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        if max_lag >= len(ac):
            continue
        sub = ac[min_lag:max_lag]
        if len(sub) == 0:
            continue
        peak = int(np.argmax(sub)) + min_lag
        if ac[0] > 0 and ac[peak] > cfg.autocorr_voiced_thr * ac[0]:
            out[i] = sr / peak
    return out


@dataclass
class ToneVADResult:
    is_speech:      bool
    voiced_ratio:   float
    voiced_seconds: float
    log_f0_std:     float
    dyn_range_oct:  float
    reason:         str
    # heuristic confidence (0..1) for showing in UI
    confidence:     float


class ToneVAD:
    """Pre-filter for the cascade: rejects noise / silence, accepts tonal speech."""

    def __init__(self, cfg: Optional[ToneVADConfig] = None):
        self.cfg = cfg or ToneVADConfig()

    def is_likely_speech(self, audio: np.ndarray, sr: int) -> ToneVADResult:
        cfg = self.cfg
        # Resample if needed
        if sr != cfg.sr_target:
            n = int(round(len(audio) * cfg.sr_target / sr))
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
            sr = cfg.sr_target

        if len(audio) < int(sr * 0.05):
            return ToneVADResult(False, 0.0, 0.0, 0.0, 0.0,
                                  "audio too short", 0.0)

        f0 = _autocorr_f0(audio, sr, cfg)
        if len(f0) == 0:
            return ToneVADResult(False, 0.0, 0.0, 0.0, 0.0,
                                  "no frames", 0.0)

        voiced_mask = (f0 > 0)
        voiced_ratio = float(voiced_mask.mean())
        voiced_seconds = float(voiced_mask.sum() * cfg.hop_ms / 1000.0)
        voiced = f0[voiced_mask]
        if len(voiced) >= 4:
            log_v = np.log(voiced + 1e-6)
            log_std = float(log_v.std())
            dyn_range_oct = float((log_v.max() - log_v.min()) / np.log(2))
        else:
            log_std = 0.0
            dyn_range_oct = 0.0

        # Decision logic
        if voiced_ratio < cfg.min_voiced_ratio:
            return ToneVADResult(False, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                                  f"voiced ratio {voiced_ratio:.1%} < {cfg.min_voiced_ratio:.0%}",
                                  voiced_ratio / max(cfg.min_voiced_ratio, 1e-6) * 0.5)
        if voiced_seconds < cfg.min_voiced_seconds:
            return ToneVADResult(False, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                                  f"voiced duration {voiced_seconds:.2f}s < {cfg.min_voiced_seconds}s",
                                  voiced_seconds / max(cfg.min_voiced_seconds, 1e-6) * 0.5)
        if dyn_range_oct > cfg.max_dyn_range_octaves:
            return ToneVADResult(False, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                                  f"F0 range {dyn_range_oct:.1f} oct > {cfg.max_dyn_range_octaves} (likely music/noise)",
                                  0.3)
        if cfg.require_tonal_pattern:
            if log_std < cfg.log_f0_std_min:
                return ToneVADResult(False, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                                      f"F0 too flat (log_std {log_std:.3f} < {cfg.log_f0_std_min}) — likely hum",
                                      0.4)
            if log_std > cfg.log_f0_std_max:
                return ToneVADResult(False, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                                      f"F0 too erratic (log_std {log_std:.3f} > {cfg.log_f0_std_max}) — likely music",
                                      0.3)

        # All gates passed
        score_voiced  = min(1.0, voiced_ratio / 0.4)
        score_dur     = min(1.0, voiced_seconds / 1.0)
        score_tonal   = 1.0 - abs(log_std - 0.20) / 0.25
        score_tonal   = max(0.0, min(1.0, score_tonal))
        confidence = 0.4 * score_voiced + 0.3 * score_dur + 0.3 * score_tonal
        return ToneVADResult(True, voiced_ratio, voiced_seconds, log_std, dyn_range_oct,
                              "tonal speech detected", confidence)


# Self-test
if __name__ == "__main__":
    import sys
    import soundfile as sf
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vad = ToneVAD()
    print("--- synthetic tests ---")
    rng = np.random.RandomState(0)
    cases = [
        ("white noise 1s",  rng.normal(0, 0.1, 16000).astype(np.float32)),
        ("silence 1s",      np.zeros(16000, dtype=np.float32)),
        ("low hum 200Hz",   (0.3 * np.sin(2 * np.pi * 200 * np.arange(16000)/16000)).astype(np.float32)),
        ("sweep 100-300Hz", (0.3 * np.sin(2 * np.pi * np.linspace(100, 300, 16000) * np.arange(16000)/16000)).astype(np.float32)),
    ]
    for name, audio in cases:
        r = vad.is_likely_speech(audio, 16000)
        print(f"  {name:<22}  speech={r.is_speech}  voiced={r.voiced_ratio:.2f}  log_std={r.log_f0_std:.3f}  {r.reason}")

    # Real Vi audio
    try:
        from pathlib import Path
        wav = "data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav"
        if Path(wav).exists():
            audio, sr = sf.read(wav, dtype="float32")
            r = vad.is_likely_speech(audio, sr)
            print(f"\n--- real Vi audio (VIVOSDEV01_R002) ---")
            print(f"  speech={r.is_speech}  voiced={r.voiced_ratio:.2f}  log_std={r.log_f0_std:.3f}  conf={r.confidence:.2f}")
            print(f"  reason: {r.reason}")
    except Exception as e:
        print(f"  [skipped real test: {e}]")
