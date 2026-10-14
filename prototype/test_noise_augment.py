"""Unit tests for the noise-augmentation core of exp/exp_noise_robustness.py.

The SNR math is the load-bearing part of the whole noise-robustness table: if
the mixture is not actually at the requested SNR, every WER number in that table
is mislabelled and the "does GTCRN help" conclusion is unfalsifiable. These
tests pin the math, the determinism, and the speech-active reference level.

No model is loaded here - the augmentation module is pure numpy on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp"))

from exp_noise_robustness import (  # noqa: E402
    NOISE_GENERATORS,
    make_noise,
    measure_snr,
    mix_at_snr,
    si_sdr,
    speech_active_mask,
    speech_active_rms,
    stable_hash,
)

SR = 16_000
ALL_TYPES = sorted(NOISE_GENERATORS)


def synthetic_speech(seconds: float = 2.0, sr: int = SR, seed: int = 1) -> np.ndarray:
    """A voiced-ish signal: harmonic stack with a syllabic envelope, unit-ish level."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    f0 = 130.0
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.283))
              for k in range(1, 9))
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    return (0.3 * sig * (0.2 + 0.8 * env)).astype(np.float32)


def speech_with_silence(speech_s: float = 1.0, silence_s: float = 9.0,
                        sr: int = SR) -> np.ndarray:
    """Short speech burst padded with a long silence - the whole-file-RMS trap."""
    speech = synthetic_speech(speech_s, sr)
    return np.concatenate([speech, np.zeros(int(silence_s * sr), dtype=np.float32)])


# --------------------------------------------------------------------------
# SNR math
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ALL_TYPES)
@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0, 15.0])
def test_requested_snr_is_achieved(kind, snr_db):
    speech = synthetic_speech()
    noise = make_noise(kind, speech.size, SR, seed=7)
    result = mix_at_snr(speech, noise, snr_db, SR)
    assert result["measured_snr_db"] == pytest.approx(snr_db, abs=0.05)


@pytest.mark.parametrize("snr_db", [0.0, 10.0])
def test_measured_snr_recomputed_from_mixture(snr_db):
    """Independent check: recover the noise as mixture - speech and re-measure."""
    speech = synthetic_speech()
    noise = make_noise("pink", speech.size, SR, seed=11)
    result = mix_at_snr(speech, noise, snr_db, SR)
    recovered = result["mixture"] - result["speech"]
    assert measure_snr(result["speech"], recovered, SR) == pytest.approx(snr_db, abs=0.05)


def test_lower_snr_means_more_noise():
    speech = synthetic_speech()
    noise = make_noise("pink", speech.size, SR, seed=3)
    loud = mix_at_snr(speech, noise, 0.0, SR)
    quiet = mix_at_snr(speech, noise, 15.0, SR)
    assert np.sqrt(np.mean(loud["noise"] ** 2)) > np.sqrt(np.mean(quiet["noise"] ** 2))


def test_peak_limiting_preserves_snr():
    """Peak limiting scales speech and noise together, so the SNR must not move."""
    speech = synthetic_speech() * 3.0  # forces the mixture past the peak limit
    noise = make_noise("pink", speech.size, SR, seed=5)
    result = mix_at_snr(speech, noise, 0.0, SR)
    assert result["peak_limited"] is True
    assert np.max(np.abs(result["mixture"])) <= 0.99 + 1e-6
    assert result["measured_snr_db"] == pytest.approx(0.0, abs=0.05)


def test_noise_is_tiled_when_shorter_than_speech():
    speech = synthetic_speech(2.0)
    short = make_noise("pink", SR // 2, SR, seed=13)
    result = mix_at_snr(speech, short, 10.0, SR)
    assert result["mixture"].size == speech.size
    assert result["measured_snr_db"] == pytest.approx(10.0, abs=0.05)


# --------------------------------------------------------------------------
# Speech-active RMS, not whole-file RMS
# --------------------------------------------------------------------------

def test_speech_active_rms_ignores_trailing_silence():
    padded = speech_with_silence(1.0, 9.0)
    whole_file = float(np.sqrt(np.mean(padded.astype(np.float64) ** 2)))
    active = speech_active_rms(padded, SR)
    # 1 s of speech in 10 s of file: whole-file RMS is ~sqrt(10) lower.
    assert active > whole_file * 2.5


def test_mix_uses_active_rms_not_whole_file_rms():
    """The discriminating test: a whole-file-RMS implementation fails this.

    With 1 s of speech in a 10 s file, whole-file RMS is ~10 dB below the
    speech-active RMS, so a whole-file implementation would inject ~10 dB less
    noise than requested and the file would sit near 20 dB while labelled 10 dB.
    """
    padded = speech_with_silence(1.0, 9.0)
    noise = make_noise("pink", padded.size, SR, seed=17)
    result = mix_at_snr(padded, noise, 10.0, SR)

    assert result["measured_snr_db"] == pytest.approx(10.0, abs=0.05)

    whole_speech = float(np.sqrt(np.mean(result["speech"].astype(np.float64) ** 2)))
    whole_noise = float(np.sqrt(np.mean(result["noise"].astype(np.float64) ** 2)))
    whole_file_snr = 20.0 * np.log10(whole_speech / whole_noise)
    assert whole_file_snr < 10.0 - 4.0  # the two references genuinely differ


def test_active_mask_selects_the_burst_region():
    padded = speech_with_silence(1.0, 9.0)
    mask = speech_active_mask(padded, SR)
    assert mask[:SR].mean() > 0.8
    assert mask[2 * SR:].mean() == 0.0


def test_active_fraction_reported():
    result = mix_at_snr(speech_with_silence(1.0, 3.0),
                        make_noise("pink", int(4 * SR), SR, seed=19), 5.0, SR)
    assert 0.1 < result["active_fraction"] < 0.5


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ALL_TYPES)
def test_generator_deterministic_from_seed(kind):
    a = make_noise(kind, SR, SR, seed=42)
    b = make_noise(kind, SR, SR, seed=42)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_generator_varies_with_seed(kind):
    a = make_noise(kind, SR, SR, seed=42)
    b = make_noise(kind, SR, SR, seed=43)
    assert not np.array_equal(a, b)


def test_mixture_deterministic():
    speech = synthetic_speech()
    first = mix_at_snr(speech, make_noise("babble", speech.size, SR, 8), 5.0, SR)
    second = mix_at_snr(speech, make_noise("babble", speech.size, SR, 8), 5.0, SR)
    assert np.array_equal(first["mixture"], second["mixture"])


def test_stable_hash_is_process_independent():
    """builtin hash() is PYTHONHASHSEED-salted; the seed derivation must not be."""
    assert stable_hash("pink") == 744714710  # golden value; crc32 is stable across processes and versions
    assert stable_hash("babble") != stable_hash("clank")


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_generators_are_unit_rms_and_finite(kind):
    x = make_noise(kind, SR, SR, seed=2)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
    assert float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) == pytest.approx(1.0, abs=1e-4)


def test_noise_types_are_spectrally_distinct():
    """Guards against a copy-paste that makes two generators the same signal."""
    def low_band_share(x):
        spectrum = np.abs(np.fft.rfft(x.astype(np.float64)))
        freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
        power = spectrum ** 2
        return float(power[freqs < 700].sum() / power.sum())

    shares = {k: low_band_share(make_noise(k, 4 * SR, SR, seed=4)) for k in ALL_TYPES}
    assert shares["hum"] > 0.8          # harmonic stack sits under 700 Hz
    assert shares["clank"] < 0.2        # metallic ring is high-frequency
    assert shares["pink"] > shares["clank"]


def test_clank_is_impulsive_and_pink_is_not():
    """Crest factor separates transient noise from stationary noise."""
    def crest(x):
        return float(np.max(np.abs(x)) / np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    assert crest(make_noise("clank", 4 * SR, SR, seed=6)) > \
           crest(make_noise("pink", 4 * SR, SR, seed=6))


# --------------------------------------------------------------------------
# Edge cases - these must raise, never silently mislabel the SNR
# --------------------------------------------------------------------------

def test_all_silence_input_raises():
    silence = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError, match="no speech-active region"):
        mix_at_snr(silence, make_noise("pink", SR, SR, seed=1), 10.0, SR)


def test_zero_length_speech_raises():
    with pytest.raises(ValueError, match="empty speech"):
        mix_at_snr(np.zeros(0, dtype=np.float32),
                   make_noise("pink", SR, SR, seed=1), 10.0, SR)


def test_zero_length_noise_raises():
    with pytest.raises(ValueError, match="empty noise"):
        mix_at_snr(synthetic_speech(), np.zeros(0, dtype=np.float32), 10.0, SR)


def test_zero_energy_noise_raises():
    with pytest.raises(ValueError, match="zero energy"):
        mix_at_snr(synthetic_speech(), np.zeros(SR * 2, dtype=np.float32), 10.0, SR)


def test_generators_handle_zero_length():
    for kind in ALL_TYPES:
        assert make_noise(kind, 0, SR, seed=1).size == 0


def test_unknown_noise_type_raises():
    with pytest.raises(ValueError, match="unknown noise type"):
        make_noise("jackhammer", SR, SR, seed=1)


def test_active_mask_on_empty_and_silent_audio():
    assert speech_active_mask(np.zeros(0, dtype=np.float32), SR).size == 0
    assert not speech_active_mask(np.zeros(SR, dtype=np.float32), SR).any()
    assert speech_active_rms(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_active_mask_shorter_than_one_frame():
    tiny = synthetic_speech(0.005)
    assert speech_active_mask(tiny, SR).all()


# --------------------------------------------------------------------------
# SI-SDR, the metric the denoiser's enhancement is reported in
# --------------------------------------------------------------------------

def test_si_sdr_is_scale_invariant():
    ref = synthetic_speech()
    noisy = mix_at_snr(ref, make_noise("pink", ref.size, SR, 21), 5.0, SR)["mixture"]
    assert si_sdr(ref, noisy) == pytest.approx(si_sdr(ref, noisy * 7.3), abs=1e-6)


def test_si_sdr_perfect_estimate_is_infinite():
    ref = synthetic_speech()
    assert si_sdr(ref, ref) == float("inf")


def test_si_sdr_decreases_with_more_noise():
    ref = synthetic_speech()
    noise = make_noise("pink", ref.size, SR, seed=23)
    at0 = si_sdr(ref, mix_at_snr(ref, noise, 0.0, SR)["mixture"])
    at15 = si_sdr(ref, mix_at_snr(ref, noise, 15.0, SR)["mixture"])
    assert at15 > at0


def test_si_sdr_empty_input_is_nan():
    assert np.isnan(si_sdr(np.zeros(0), np.zeros(0)))
