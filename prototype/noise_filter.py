"""
Spectral-subtraction noise filter — pre-ASR audio enhancement.

Industrial environments (factories, construction, hospitals) produce
broadband ambient noise (fans, machinery, HVAC). Generic ASR models trained
on clean speech degrade significantly under such noise. We pre-filter the
audio with a cheap spectral-subtraction (Boll 1979) before passing it to
the cascade.

Paper §3.X candidate: typology-naive noise filter that COMPOSES with our
tonal-perceptual cache (no fingerprint change needed — cache key reflects
the filtered audio, not the raw input).

Algorithm:
  1. Estimate noise PSD from the first NOISE_EST_MS of audio (assumed
     silence/background).
  2. Compute STFT of full signal.
  3. Subtract scaled noise magnitude per frequency bin.
  4. Apply spectral-floor clamp to avoid musical noise artifacts.
  5. Inverse STFT.

Latency: ~3-5 ms on a 4 s utterance @ 16 kHz on CPU. Negligible vs cascade
latency (5-7 s).

Self-test: synthetic clean speech + injected white noise; measure SNR before
and after filtering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NoiseFilterConfig:
    sr:                int   = 16000
    n_fft:             int   = 512
    hop:               int   = 160
    noise_est_ms:      float = 300.0      # estimate noise from first N ms of input
    subtraction_alpha: float = 1.5        # over-subtraction factor (Boll 1979 sets ~2)
    spectral_floor:    float = 0.05       # min residual after subtraction (avoid zeros)
    smoothing_beta:    float = 0.85       # EMA on noise estimate across frames


@dataclass
class NoiseFilterResult:
    filtered_audio:  np.ndarray
    sr:              int
    snr_in_db_est:   float
    snr_out_db_est:  float
    noise_est_seconds: float


class NoiseFilter:
    """Stateless spectral subtraction. One call per utterance."""

    def __init__(self, cfg: Optional[NoiseFilterConfig] = None):
        self.cfg = cfg or NoiseFilterConfig()

    def __call__(self, audio: np.ndarray, sr: int) -> NoiseFilterResult:
        cfg = self.cfg
        # Resample if needed
        if sr != cfg.sr:
            n = int(round(len(audio) * cfg.sr / sr))
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
            sr = cfg.sr

        if len(audio) < cfg.n_fft * 2:
            return NoiseFilterResult(audio, sr, 0.0, 0.0, 0.0)

        # Auto-bypass: if the leading noise estimate is below a tiny absolute
        # floor, the signal is already clean — spectral subtraction would just
        # add musical noise. Skip the filter and return the input unchanged.
        n_lead = min(int(sr * cfg.noise_est_ms / 1000), len(audio))
        lead_rms = float(np.sqrt((audio[:n_lead] ** 2).mean() + 1e-12))
        if lead_rms < 0.005:    # ~ -46 dBFS leading silence → input is clean
            est_snr = 20.0 * np.log10(max(float(np.sqrt((audio ** 2).mean() + 1e-12)) / max(lead_rms, 1e-9), 1.0))
            return NoiseFilterResult(audio.astype(np.float32), sr, est_snr, est_snr, cfg.noise_est_ms / 1000)

        # ---- STFT ----
        n_noise_samples = int(sr * cfg.noise_est_ms / 1000)
        n_frames = (len(audio) - cfg.n_fft) // cfg.hop + 1
        window = np.hanning(cfg.n_fft).astype(np.float32)
        frames = np.stack([audio[i * cfg.hop:i * cfg.hop + cfg.n_fft] * window
                            for i in range(n_frames)], axis=0)
        spec = np.fft.rfft(frames, axis=1)
        mag  = np.abs(spec)
        phase = np.angle(spec)

        # ---- Noise PSD estimate from leading frames ----
        n_noise_frames = max(1, n_noise_samples // cfg.hop)
        noise_mag = mag[:n_noise_frames].mean(axis=0)   # (n_fft//2+1,)
        snr_in_est_lin = (mag.mean() / (noise_mag.mean() + 1e-9))
        snr_in_db = 20.0 * np.log10(max(snr_in_est_lin, 1e-6))

        # ---- Over-subtraction with spectral floor ----
        # Update noise estimate via EMA on low-power frames (silence detection)
        frame_pow = (mag ** 2).mean(axis=1)
        avg_pow = float(np.mean(frame_pow))
        for i in range(n_frames):
            if frame_pow[i] < 0.3 * avg_pow:
                noise_mag = cfg.smoothing_beta * noise_mag + (1 - cfg.smoothing_beta) * mag[i]

        clean_mag = np.maximum(
            mag - cfg.subtraction_alpha * noise_mag[None, :],
            cfg.spectral_floor * mag,
        )
        snr_out_est_lin = (clean_mag.mean() / (noise_mag.mean() + 1e-9))
        snr_out_db = 20.0 * np.log10(max(snr_out_est_lin, 1e-6))

        # ---- Inverse STFT ----
        clean_spec = clean_mag * np.exp(1j * phase)
        clean_frames = np.fft.irfft(clean_spec, axis=1)
        # Overlap-add
        out = np.zeros(len(audio), dtype=np.float32)
        norm = np.zeros(len(audio), dtype=np.float32)
        for i in range(n_frames):
            s = i * cfg.hop
            e = s + cfg.n_fft
            if e > len(out): break
            out[s:e]  += clean_frames[i] * window
            norm[s:e] += window ** 2
        norm = np.where(norm > 1e-6, norm, 1.0)
        out  = out / norm

        return NoiseFilterResult(
            filtered_audio=out.astype(np.float32),
            sr=sr,
            snr_in_db_est=float(snr_in_db),
            snr_out_db_est=float(snr_out_db),
            noise_est_seconds=cfg.noise_est_ms / 1000,
        )


# Self-test on real VIVOS audio + injected noise
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import soundfile as sf

    wav = Path("data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav")
    if not wav.exists():
        print("[skip] no VIVOS audio")
        sys.exit(0)

    clean, sr = sf.read(str(wav), dtype="float32")
    rng = np.random.default_rng(0)
    nf = NoiseFilter()

    print("--- noise filter spectral-subtraction self-test ---")
    for noise_amp in [0.0, 0.01, 0.05, 0.10, 0.20]:
        noisy = clean + rng.normal(0, noise_amp, len(clean)).astype(np.float32)
        t0 = time.perf_counter()
        r = nf(noisy, sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Real SNR (we know the noise we added)
        noise_power = noise_amp ** 2
        clean_power = float((clean ** 2).mean())
        real_snr_in = 10.0 * np.log10(clean_power / max(noise_power, 1e-12))
        residual = r.filtered_audio - clean
        real_snr_out = 10.0 * np.log10(clean_power / max(float((residual**2).mean()), 1e-12))
        print(f"  noise σ={noise_amp:.2f}  real SNR_in={real_snr_in:6.1f} dB  "
              f"real SNR_out={real_snr_out:6.1f} dB  Δ={real_snr_out-real_snr_in:+5.1f} dB  "
              f"filter_ms={elapsed_ms:5.1f}")
