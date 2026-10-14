"""Industrial-noise robustness of the ASR front-end: does GTCRN actually help?

Risk register (CLAUDE.md section 7) lists "Industrial noise destroys ASR" at
Likelihood High / Impact High, and the competition is judged on factories,
construction sites, logistics hubs and hospitals. We ship a GTCRN denoiser and a
Silero VAD in `prototype/edge_frontend.py` and have never measured what either
one buys under noise: every accuracy number we hold is on clean read speech.
This script produces that missing evidence.

  Experiment N1  WER/CER on VIVOS in three conditions - clean, noisy, and
                 noisy-with-GTCRN - across a noise-type x SNR grid.
  Experiment N2  The enhancement the denoiser actually achieves, as SI-SDR
                 improvement measured here, plus its latency cost (p50/p95).
  Experiment N3  Silero VAD false-trigger rate on noise-only segments. A VAD
                 that fires on machine hum spends a whole ASR pass on nothing;
                 that is a stability defect, not merely an accuracy one.

*** TWO NOISE SOURCES - READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT ***

`noise_source` in the config selects which one runs; "both" runs the grid twice,
once per source, and every cell in the JSON carries its own `noise_source` and
`noise_source_caveat` so a number can never be quoted against the wrong source.

  "corpus"     REAL recorded noise from the DEMAND corpus (Thiemann, Ito &
               Vincent 2013, doi:10.5281/zenodo.1227121, CC BY 4.0). CLAUDE.md
               section 4 prescribes PSTATION + DKITCHEN; we ship those plus
               PCAFETER, whose real multi-talker babble is the direct
               real-corpus counterpart of the synthetic `babble` cell where the
               denoiser was found to be significantly harmful. One channel
               (ch01) of the official 16 kHz release, 5 minutes per environment.
               Segments are drawn at a seed-determined offset, so a cell is
               reproducible given the dataset. Provenance (file, offset,
               sha256 of the source wav) is recorded per utterance.

  "synthetic"  Noise generated deterministically from a seed: pink broadband,
               machine hum with harmonics, impulsive metallic clanks,
               multi-talker babble approximation. Kept because it needs no
               download and is bit-reproducible anywhere, and because the
               already-published `noise-t4-v1` grid used it - deleting it would
               strand those numbers with nothing to compare against. It is NOT
               DEMAND and NOT MUSAN; synthetic numbers are valid for RELATIVE
               comparison inside a run and are NOT comparable to any published
               WER on DEMAND/MUSAN-augmented data.

MUSAN (CLAUDE.md also prescribes it) is NOT wired in. It is distributed only as
a single 11 GB tarball (openslr.org/17, CC BY 4.0) with no official subset, and
the third-party 50-file slices on the Hub have no verifiable provenance. Adding
it means fetching the full tarball; until someone does, there is no MUSAN
condition here and none should be claimed.

Runs both locally on CPU (small n, for iteration) and as a Kaggle T4 kernel
(full n); device and sample counts come from config exactly as
`exp_t4_bidirectional.py` does it.

Local:
    python exp/exp_noise_robustness.py --samples 8 --snrs 0,10 --types pink,hum
    python exp/exp_noise_robustness.py --samples 2 --snrs 5 --noise-source corpus \\
        --corpus-types PSTATION --no-vad
Kaggle kernel requirements: GPU T4, Internet disabled, dataset inputs
    daosyduyminh/voiceai-edge-models-t4
    daosyduyminh/voiceai-saigonhub-tier-1-speech-vi-zh-ko-en
    daosyduyminh/voiceai-t4-srclib
    daosyduyminh/voiceai-t4-wheels-py312
    daosyduyminh/voiceai-t4-noise-config   (optional; overrides DEFAULT_CONFIG)
    daosyduyminh/voiceai-t4-noise-demand   (required when noise_source != synthetic)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()

SYNTHETIC_NOISE_DISCLAIMER = (
    "SYNTHETIC NOISE, NOT DEMAND/MUSAN. The noise in this cell is generated "
    "deterministically from a seed, not recorded. Results are valid for "
    "relative comparison within this run only and are NOT comparable to "
    "published numbers on DEMAND or MUSAN. Run with noise_source='corpus' for "
    "the real-corpus (DEMAND) condition."
)

CORPUS_NOISE_DISCLAIMER = (
    "REAL RECORDED NOISE from the DEMAND corpus (Thiemann, Ito & Vincent 2013, "
    "doi:10.5281/zenodo.1227121, CC BY 4.0), official 16 kHz release, channel "
    "ch01, segment drawn at a seed-determined offset. This satisfies the DEMAND "
    "half of CLAUDE.md section 4. MUSAN is still NOT included - there is no "
    "official MUSAN subset, only an 11 GB tarball - so no MUSAN claim may be "
    "made from this run."
)

NOISE_SOURCE_CAVEATS = {
    "synthetic": SYNTHETIC_NOISE_DISCLAIMER,
    "demand": CORPUS_NOISE_DISCLAIMER,
}

DEFAULT_CONFIG = {
    "condition": "noise-local-smoke-v1",
    "samples": 8,
    "warmup_samples": 1,
    "snr_db": [0, 5, 10, 15],
    "noise_source": "synthetic",          # synthetic | corpus | both
    "noise_types": ["pink", "hum", "clank", "babble"],
    "corpus_noise_types": ["PSTATION", "DKITCHEN", "PCAFETER"],
    "corpus_noise_dataset": "voiceai-t4-noise-demand",
    "corpus_noise_dir": "",               # explicit path wins over the lookup
    "seed": 20260820,
    "sample_rate": 16_000,
    "device": "cpu",
    "asr_compute_type": "int8",
    "required_gpu_substring": "",
    "vad_trials_per_cell": 12,
    "vad_segment_s": 3.0,
    "asr_model": "phowhisper-small-int8",
}

KAGGLE_CONFIG_OVERRIDE = {
    "condition": "noise-t4-v1",
    "samples": 100,
    "warmup_samples": 2,
    "device": "cuda",
    "asr_compute_type": "int8_float16",
    "required_gpu_substring": "T4",
    "vad_trials_per_cell": 24,
}


# ==========================================================================
# Noise augmentation - the importable, unit-testable core (see
# prototype/test_noise_augment.py). Nothing below this banner imports a model,
# so the tests run with numpy alone.
# ==========================================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)


def stable_hash(text: str) -> int:
    """Deterministic across processes - builtin hash() is salted by PYTHONHASHSEED
    and would silently give a different noise realization on every run."""
    return zlib.crc32(text.encode("utf-8"))


def _unit_rms(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if r <= 0.0:
        return x.astype(np.float32)
    return (x / r).astype(np.float32)


def pink_noise(n: int, sr: int, seed: int) -> np.ndarray:
    """Broadband stationary noise with a 1/f magnitude tilt.

    Stands in for the wide-band hiss of an air handler or an open shop floor:
    it raises the masking floor uniformly across the spectrum, which is the
    classic way stationary noise removes low-energy consonants.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tilt = np.ones_like(freqs)
    tilt[1:] = 1.0 / np.sqrt(freqs[1:])
    tilt[0] = tilt[1] if freqs.size > 1 else 1.0
    return _unit_rms(np.fft.irfft(spectrum * tilt, n=n))


def machine_hum(n: int, sr: int, seed: int, fundamental: float = 50.0,
                n_harmonics: int = 12) -> np.ndarray:
    """Low-frequency motor hum: a mains-rate fundamental plus decaying harmonics.

    Attacks ASR differently from pink noise - the energy is concentrated below
    ~700 Hz where F0 and F1 live, which is precisely the band Vietnamese tone
    contours occupy, so this is the generator to watch for tone damage.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        f = fundamental * k
        if f >= sr / 2:
            break
        amp = 1.0 / (k ** 1.2)
        wobble = 1.0 + 0.02 * np.sin(2 * np.pi * rng.uniform(0.1, 0.7) * t + rng.uniform(0, 6.283))
        out += amp * np.sin(2 * np.pi * f * t * wobble + rng.uniform(0, 6.283))
    out += 0.05 * rng.standard_normal(n)  # bearing/brush broadband floor
    return _unit_rms(out)


def impulsive_clank(n: int, sr: int, seed: int, events_per_second: float = 2.0) -> np.ndarray:
    """Sparse metallic transients - dropped tooling, press strikes, pallet drops.

    Non-stationary by construction. Any denoiser tuned on stationary noise
    tends to leave these through, and they corrupt onsets rather than the
    steady state, so they break ASR through a different mechanism than hiss.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _rng(seed)
    out = np.zeros(n, dtype=np.float64)
    n_events = max(1, int(round(events_per_second * n / sr)))
    for _ in range(n_events):
        start = int(rng.integers(0, n))
        dur = int(sr * rng.uniform(0.05, 0.25))
        idx = np.arange(min(dur, n - start), dtype=np.float64)
        if idx.size == 0:
            continue
        env = np.exp(-idx / (sr * rng.uniform(0.01, 0.05)))
        ring = np.zeros_like(idx)
        for partial in rng.uniform(1800.0, 6500.0, size=4):
            ring += np.sin(2 * np.pi * partial * idx / sr + rng.uniform(0, 6.283))
        out[start:start + idx.size] += env * ring * rng.uniform(0.5, 1.5)
    return _unit_rms(out)


def babble(n: int, sr: int, seed: int, n_talkers: int = 8) -> np.ndarray:
    """Multi-talker babble approximation - speech-shaped, syllable-modulated.

    Each talker is a set of formant-band-filtered noise carriers amplitude
    modulated at a 2-6 Hz syllabic rate. This is informational masking: the
    interferer occupies the same spectro-temporal cells as the target, which is
    the condition denoisers handle worst because it is not spectrally separable.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rng = _rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    out = np.zeros(n, dtype=np.float64)
    for _ in range(n_talkers):
        shape = np.zeros_like(freqs)
        for center, width in ((rng.uniform(250, 450), 120.0),
                              (rng.uniform(900, 1600), 350.0),
                              (rng.uniform(2200, 3200), 600.0)):
            shape += np.exp(-0.5 * ((freqs - center) / width) ** 2)
        shape *= np.exp(-freqs / 4000.0)  # long-term average speech spectrum tilt
        carrier = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n=n)
        rate = rng.uniform(2.0, 6.0)
        env = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.283)))
        out += carrier * (0.15 + 0.85 * env ** 2)
    return _unit_rms(out)


NOISE_GENERATORS = {
    "pink": pink_noise,
    "hum": machine_hum,
    "clank": impulsive_clank,
    "babble": babble,
}

NOISE_DESCRIPTIONS = {
    "pink": "synthetic broadband stationary 1/f noise (NOT DEMAND PSTATION)",
    "hum": "synthetic 50 Hz machine hum + 12 harmonics (NOT DEMAND PSTATION)",
    "clank": "synthetic sparse impulsive metallic transients (NOT MUSAN)",
    "babble": "synthetic 8-talker speech-shaped babble (NOT MUSAN speech)",
}


def make_noise(kind: str, n: int, sr: int, seed: int) -> np.ndarray:
    if kind not in NOISE_GENERATORS:
        raise ValueError(f"unknown noise type {kind!r}; have {sorted(NOISE_GENERATORS)}")
    return NOISE_GENERATORS[kind](n, sr, seed)


# ==========================================================================
# Real-corpus noise source (DEMAND). This is a noise SOURCE only - it hands
# `mix_at_snr` a unit-RMS waveform exactly like the synthetic generators do, so
# the mixing math (speech-active-RMS referenced, verified to 0.05 dB by
# prototype/test_noise_augment.py) is untouched and the two sources stay
# directly comparable.
# ==========================================================================

_CORPUS_AUDIO_CACHE: dict = {}
_CORPUS_CONTEXT: dict = {}


def corpus_noise_root(cfg: dict) -> Path:
    explicit = str(cfg.get("corpus_noise_dir") or "")
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise FileNotFoundError(f"corpus_noise_dir does not exist: {root}")
        return root
    if ON_KAGGLE:
        slug = str(cfg.get("corpus_noise_dataset") or "voiceai-t4-noise-demand")
        return unwrap_same_name(dataset_root(slug))
    return PROJECT / "data" / "raw" / "noise_corpus"


def load_corpus_manifest(root: Path) -> dict:
    """Read the manifest that travels with the noise corpus.

    The manifest, not this script, is the record of what the audio actually is -
    corpus, licence, channel, sample rate, and the sha256 of every file. It is
    copied verbatim into the JSON so a result can always be traced back to the
    bytes that produced it."""
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No manifest.json under {root}. The real-corpus noise source refuses to "
            f"run on an undocumented pile of wav files - see kaggle_uploads/"
            f"voiceai-t4-noise-demand/README.md for how the corpus is built.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("environments"):
        raise ValueError(f"manifest at {path} lists no environments")
    return manifest


def _load_corpus_env(root: Path, manifest: dict, env: str, sr: int):
    key = (str(root), env)
    if key in _CORPUS_AUDIO_CACHE:
        return _CORPUS_AUDIO_CACHE[key]
    entry = manifest["environments"].get(env)
    if entry is None:
        raise ValueError(f"environment {env!r} not in manifest; have "
                         f"{sorted(manifest['environments'])}")
    import soundfile as sf
    path = Path(root) / entry["file"]
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(file_sr) != int(sr):
        # Deliberately fatal rather than resampled here: the corpus is shipped
        # already at the benchmark rate, so a mismatch means the wrong dataset
        # is mounted, and silently resampling would hide that.
        raise ValueError(f"{path} is {file_sr} Hz, benchmark runs at {sr} Hz")
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    _CORPUS_AUDIO_CACHE[key] = (audio, entry)
    return audio, entry


def corpus_noise(env: str, n: int, sr: int, seed: int, root: Path,
                 manifest: dict) -> tuple:
    """One unit-RMS noise segment cut from a real DEMAND recording.

    The offset is drawn from the same seed the synthetic generators get, so the
    cell is reproducible given the dataset, and the offset is reported so any
    single utterance can be regenerated by hand."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32), {"environment": env, "reason": "empty request"}
    audio, entry = _load_corpus_env(root, manifest, env, sr)
    rng = _rng(seed)
    if audio.size >= n:
        offset = int(rng.integers(0, audio.size - n + 1))
        segment = audio[offset:offset + n]
        wrapped = False
    else:
        offset = int(rng.integers(0, audio.size))
        segment = np.take(audio, (np.arange(n) + offset) % audio.size)
        wrapped = True
    provenance = {
        "environment": env,
        "file": entry["file"],
        "sha256": entry.get("sha256"),
        "offset_samples": offset,
        "offset_s": offset / float(sr),
        "length_s": n / float(sr),
        "wrapped": wrapped,
    }
    return _unit_rms(np.ascontiguousarray(segment, dtype=np.float32)), provenance


def resolve_noise_specs(cfg: dict) -> list:
    """Expand the config into the list of (source, type) cells the grid runs.

    Every spec carries its own source label and caveat; nothing downstream ever
    infers the source from the type name."""
    source = str(cfg.get("noise_source") or "synthetic").lower()
    if source not in {"synthetic", "corpus", "both"}:
        raise ValueError(f"noise_source must be synthetic|corpus|both, got {source!r}")
    specs = []
    if source in ("synthetic", "both"):
        for kind in cfg["noise_types"]:
            if kind not in NOISE_GENERATORS:
                raise ValueError(f"unknown synthetic noise type {kind!r}; "
                                 f"have {sorted(NOISE_GENERATORS)}")
            specs.append({"noise_source": "synthetic", "noise_type": kind,
                          "cell_id": f"synthetic:{kind}",
                          "noise_description": NOISE_DESCRIPTIONS[kind],
                          "noise_source_caveat": SYNTHETIC_NOISE_DISCLAIMER})
    if source in ("corpus", "both"):
        root = corpus_noise_root(cfg)
        manifest = load_corpus_manifest(root)
        _CORPUS_CONTEXT["root"] = root
        _CORPUS_CONTEXT["manifest"] = manifest
        corpus_name = manifest.get("corpus", "DEMAND")
        for env in cfg["corpus_noise_types"]:
            entry = manifest["environments"].get(env)
            if entry is None:
                raise ValueError(f"environment {env!r} not in {root}/manifest.json; "
                                 f"have {sorted(manifest['environments'])}")
            specs.append({
                "noise_source": corpus_name.lower(),
                "noise_type": env,
                "cell_id": f"{corpus_name.lower()}:{env}",
                "noise_description": (f"REAL {corpus_name} {env} "
                                      f"({entry.get('description', 'no description')}), "
                                      f"{entry.get('channel', '?')}, "
                                      f"{manifest.get('license', 'licence unrecorded')}"),
                "noise_source_caveat": NOISE_SOURCE_CAVEATS.get(
                    corpus_name.lower(), CORPUS_NOISE_DISCLAIMER),
                "corpus_file": entry["file"],
                "corpus_sha256": entry.get("sha256"),
            })
    if not specs:
        raise ValueError("no noise cells resolved from config")
    return specs


def realize_noise(spec: dict, n: int, sr: int, seed: int) -> tuple:
    """One noise realization for a spec, plus its provenance (None if synthetic)."""
    if spec["noise_source"] == "synthetic":
        return make_noise(spec["noise_type"], n, sr, seed), None
    return corpus_noise(spec["noise_type"], n, sr, seed,
                        _CORPUS_CONTEXT["root"], _CORPUS_CONTEXT["manifest"])


def speech_active_mask(audio: np.ndarray, sr: int, frame_ms: float = 20.0,
                       hop_ms: float = 10.0, floor_db: float = 35.0) -> np.ndarray:
    """Boolean per-sample mask of the speech-active region.

    Energy-gated relative to the loudest frame: frames within `floor_db` of the
    peak frame count as active. This exists because the SNR must be referenced
    to the level of the speech itself. A VIVOS utterance with two seconds of
    leading silence has a whole-file RMS well below its speech RMS, and mixing
    against the whole-file RMS would quietly place the noise at a higher SNR
    than requested - the mistake that makes a noise-robustness table meaningless.
    """
    n = int(audio.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if n < frame:
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    starts = range(0, n - frame + 1, hop)
    energies = np.array([float(np.mean(np.square(audio[s:s + frame], dtype=np.float64)))
                         for s in starts])
    peak = float(energies.max())
    if peak <= 0.0:
        return np.zeros(n, dtype=bool)
    threshold = peak * (10.0 ** (-floor_db / 10.0))
    for s, e in zip(starts, energies):
        if e >= threshold:
            mask[s:s + frame] = True
    return mask


def speech_active_rms(audio: np.ndarray, sr: int, **kwargs) -> float:
    mask = speech_active_mask(audio, sr, **kwargs)
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio[mask], dtype=np.float64))))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, sr: int,
               peak_limit: float = 0.99) -> dict:
    """Mix `noise` into `speech` at `snr_db`, referenced to speech-active RMS.

    Returns a dict with the mixture, the scaled noise, the requested and the
    measured SNR, and whether peak limiting was applied. Raises ValueError on
    zero-length input or on audio with no speech-active region, because in
    either case there is no reference level and any mixture would carry a
    silently wrong SNR.
    """
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if speech.size == 0:
        raise ValueError("mix_at_snr: empty speech signal, no reference level")
    if noise.size == 0:
        raise ValueError("mix_at_snr: empty noise signal")
    if noise.size < speech.size:
        noise = np.tile(noise, int(np.ceil(speech.size / noise.size)))
    noise = noise[:speech.size]

    mask = speech_active_mask(speech, sr)
    if not mask.any():
        raise ValueError("mix_at_snr: no speech-active region (all-silence input)")
    s_rms = float(np.sqrt(np.mean(np.square(speech[mask], dtype=np.float64))))
    n_rms = float(np.sqrt(np.mean(np.square(noise[mask], dtype=np.float64))))
    if s_rms <= 0.0:
        raise ValueError("mix_at_snr: speech-active region has zero energy")
    if n_rms <= 0.0:
        raise ValueError("mix_at_snr: noise has zero energy over the speech region")

    scale = (s_rms / n_rms) / (10.0 ** (snr_db / 20.0))
    scaled = (noise * scale).astype(np.float32)
    mixture = (speech + scaled).astype(np.float32)

    limited, peak = False, float(np.max(np.abs(mixture))) if mixture.size else 0.0
    if peak > peak_limit:
        # Scaling both components together preserves the SNR exactly; clipping
        # would not, and a clipped mixture would confound noise damage with
        # distortion damage.
        gain = peak_limit / peak
        mixture = (mixture * gain).astype(np.float32)
        scaled = (scaled * gain).astype(np.float32)
        speech_out = (speech * gain).astype(np.float32)
        limited = True
    else:
        speech_out = speech

    return {
        "mixture": mixture,
        "noise": scaled,
        "speech": speech_out,
        "requested_snr_db": float(snr_db),
        "measured_snr_db": measure_snr(speech_out, scaled, sr),
        "peak_limited": limited,
        "peak_before_limit": peak,
        "active_fraction": float(mask.mean()),
    }


def measure_snr(speech: np.ndarray, noise: np.ndarray, sr: int) -> float:
    """SNR in dB over the speech-active region of `speech`."""
    mask = speech_active_mask(speech, sr)
    if not mask.any():
        return float("nan")
    s = float(np.sqrt(np.mean(np.square(speech[mask], dtype=np.float64))))
    n = float(np.sqrt(np.mean(np.square(noise[mask], dtype=np.float64))))
    if n <= 0.0:
        return float("inf")
    if s <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(s / n))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB, the standard enhancement metric.

    Used instead of a plain input/output SNR because GTCRN is non-linear: the
    residual `estimate - reference` is not the input noise, and the denoiser
    applies its own gain, so a naive SNR would reward or punish level changes
    that ASR does not care about. SI-SDR improvement (output minus input) is the
    quantity reported as the denoiser's enhancement, and it is NOT the same
    measurement as the "+10.4 dB" figure recorded in prototype/EDGE_UPGRADES.md.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    n = min(ref.size, est.size)
    if n == 0:
        return float("nan")
    ref, est = ref[:n], est[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float(np.dot(ref, ref))
    if denom <= 0.0:
        return float("nan")
    projection = (float(np.dot(est, ref)) / denom) * ref
    residual = est - projection
    num, den = float(np.dot(projection, projection)), float(np.dot(residual, residual))
    if den <= 0.0:
        return float("inf")
    if num <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(num / den))


# ==========================================================================
# Metrics and Kaggle layout - copied from exp_t4_bidirectional.py so the two
# benchmarks score identically. Divergence here would silently make the noise
# WER incomparable to the clean WER already reported in Results/kaggle_t4.
# ==========================================================================

def dataset_root(slug: str) -> Path:
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    owners = KAGGLE_INPUT / "datasets"
    if owners.is_dir():
        for owner in sorted(owners.iterdir()):
            candidate = owner / slug
            if candidate.is_dir():
                return candidate
    if KAGGLE_INPUT.is_dir():
        matches = sorted((p for p in KAGGLE_INPUT.glob(f"*/*/{slug}") if p.is_dir()),
                         key=lambda p: len(p.parts))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Kaggle dataset not mounted: {slug}")


def unwrap_same_name(path: Path) -> Path:
    while True:
        nested = path / path.name
        if not nested.is_dir():
            return path
        path = nested


def payload_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return unwrap_same_name(direct)
    matches = sorted((p for p in root.rglob(name) if p.is_dir()),
                     key=lambda p: len(p.parts))
    if matches:
        return unwrap_same_name(matches[0])
    raise FileNotFoundError(f"Missing payload directory {name!r} under {root}")


def select_wheels(wheel_files: list) -> list:
    abi_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    chosen: dict = {}
    for path in sorted(wheel_files):
        parts = path.name.split("-")
        if len(parts) < 5:
            continue
        dist, python_tag = parts[0].lower().replace("_", "-"), parts[2]
        pure = python_tag.startswith("py3") or python_tag.startswith("py2.py3")
        if not (pure or abi_tag in python_tag.split(".")):
            continue
        if dist not in chosen or not pure:
            chosen[dist] = path
    return [chosen[key] for key in sorted(chosen)]


def install_offline_wheels(roots: list) -> None:
    found: list = []
    for root in roots:
        found.extend(p for p in root.rglob("*.whl") if p.is_file())
    wheels = select_wheels(found)
    if not wheels:
        raise FileNotFoundError(f"No wheels for this interpreter among {[p.name for p in found]}")
    print(f"[wheels] installing {[p.name for p in wheels]}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                    "--no-deps", *map(str, wheels)], check=True)


def normalize_for_scoring(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFC. Keeps diacritics."""
    text = unicodedata.normalize("NFC", text or "").casefold()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in text]
    return " ".join("".join(kept).split())


def _levenshtein(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    previous = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        current = [i]
        for j, r in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (h != r)))
        previous = current
    return previous[-1]


def wer_cer(reference: str, hypothesis: str) -> dict:
    ref_n, hyp_n = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    if not ref_n:
        return {"wer": None, "cer": None, "ref_words": 0, "ref_chars": 0,
                "unscored": "empty reference"}
    ref_w, hyp_w = ref_n.split(), hyp_n.split()
    ref_c, hyp_c = list(ref_n.replace(" ", "")), list(hyp_n.replace(" ", ""))
    return {"wer": _levenshtein(ref_w, hyp_w) / max(len(ref_w), 1),
            "cer": _levenshtein(ref_c, hyp_c) / max(len(ref_c), 1),
            "ref_words": len(ref_w), "ref_chars": len(ref_c)}


def pct(values: list, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stat_block(values: list) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"n": 0, "reason": "no samples"}
    return {"n": len(values), "mean": statistics.fmean(values),
            "p50": pct(values, 50), "p95": pct(values, 95),
            "min": min(values), "max": max(values)}


def bootstrap_ci(values: list, seed: int, resamples: int = 1000, alpha: float = 0.05) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if len(values) < 2:
        return {"n": len(values), "reason": "too few samples for a CI"}
    arr = np.asarray(values, dtype=np.float64)
    rng = _rng(seed)
    means = arr[rng.integers(0, arr.size, size=(resamples, arr.size))].mean(axis=1)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "lo": float(np.percentile(means, 100 * alpha / 2)),
            "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
            "resamples": resamples}


# ==========================================================================
# Corpus / config
# ==========================================================================

def load_config(cli: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if ON_KAGGLE:
        cfg.update(KAGGLE_CONFIG_OVERRIDE)
    path = None
    if ON_KAGGLE:
        try:
            path = dataset_root("voiceai-t4-noise-config") / "benchmark_config.json"
        except FileNotFoundError:
            path = HERE / "noise_config.json"
    else:
        path = HERE / "noise_config.json"
    if path is not None and path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    cfg["resolved_config_path"] = str(path)
    for key in ("samples", "warmup_samples", "seed", "condition", "device",
                "asr_compute_type", "vad_trials_per_cell", "noise_source",
                "corpus_noise_dir"):
        value = getattr(cli, key, None)
        if value is not None:
            cfg[key] = value
    if cli.snrs:
        cfg["snr_db"] = [float(s) for s in cli.snrs.split(",") if s.strip()]
    if cli.types:
        cfg["noise_types"] = [s.strip() for s in cli.types.split(",") if s.strip()]
    if cli.corpus_types:
        cfg["corpus_noise_types"] = [s.strip() for s in cli.corpus_types.split(",")
                                     if s.strip()]
    return cfg


def vivos_root() -> Path:
    if ON_KAGGLE:
        return payload_dir(dataset_root("voiceai-saigonhub-tier-1-speech-vi-zh-ko-en"), "vivos")
    return PROJECT / "data" / "processed" / "vivos" / "vivos"


def load_vivos(root: Path, limit: int) -> list:
    prompts = {}
    for line in (root / "test" / "prompts.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            prompts[parts[0]] = parts[1]
    out = []
    for wav in sorted((root / "test" / "waves").rglob("*.wav")):
        ref = prompts.get(wav.stem)
        if ref:
            out.append({"id": wav.stem, "path": str(wav), "reference": ref})
        if len(out) >= limit:
            break
    if not out:
        raise RuntimeError(f"No VIVOS test utterances under {root}")
    return out


def read_wav(path: str, sr_target: int = 16_000) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sr_target:
        n = int(round(len(audio) * sr_target / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return np.ascontiguousarray(audio, dtype=np.float32)


def resolve_asr_dir(cfg: dict) -> Path:
    if ON_KAGGLE:
        return payload_dir(dataset_root("voiceai-edge-models-t4"), "models_ct2") / cfg["asr_model"]
    return PROJECT / "data" / "raw" / "models_ct2" / cfg["asr_model"]


# ==========================================================================
# Benchmark
# ==========================================================================

def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", type=int)
    ap.add_argument("--warmup-samples", dest="warmup_samples", type=int)
    ap.add_argument("--snrs", help="comma-separated dB, e.g. 0,5,10,15")
    ap.add_argument("--types", help=f"comma-separated synthetic types from "
                                    f"{sorted(NOISE_GENERATORS)}")
    ap.add_argument("--noise-source", dest="noise_source",
                    choices=["synthetic", "corpus", "both"],
                    help="synthetic generators, the real DEMAND corpus, or both")
    ap.add_argument("--corpus-types", dest="corpus_types",
                    help="comma-separated DEMAND environments, e.g. PSTATION,DKITCHEN")
    ap.add_argument("--corpus-noise-dir", dest="corpus_noise_dir",
                    help="directory holding manifest.json + the corpus wavs")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--condition")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    ap.add_argument("--asr-compute-type", dest="asr_compute_type")
    ap.add_argument("--vad-trials", dest="vad_trials_per_cell", type=int)
    ap.add_argument("--no-vad", action="store_true", help="skip experiment N3")
    ap.add_argument("--out", help="output JSON path (default: Results/ or /kaggle/working)")
    cli = ap.parse_args()

    cfg = load_config(cli)
    sr = int(cfg["sample_rate"])
    print(f"[config] {json.dumps({k: v for k, v in cfg.items() if k != 'resolved_config_path'})}")

    if ON_KAGGLE:
        roots = []
        for slug in ("voiceai-t4-wheels-py312", "voiceai-edge-models-t4"):
            try:
                roots.append(dataset_root(slug))
            except FileNotFoundError:
                pass
        if roots:
            install_offline_wheels(roots)
        sys.path.insert(0, str(HERE))
        try:
            srclib = dataset_root("voiceai-t4-srclib")
            sys.path.insert(0, str(srclib))
            # edge_frontend resolves its ONNX weights relative to the source tree
            # and would try to download them; on this kernel the mount is
            # read-only and there is no network. Point it at the shipped copies.
            if (srclib / "gtcrn_simple.onnx").is_file():
                os.environ["VOICEAI_FRONTEND_MODELS_DIR"] = str(srclib)
        except FileNotFoundError:
            pass
    sys.path.insert(0, str(PROJECT / "prototype"))

    noise_specs = resolve_noise_specs(cfg)
    active_sources = sorted({s["noise_source"] for s in noise_specs})
    for source in active_sources:
        print(f"[noise ] source={source}: "
              f"{[s['noise_type'] for s in noise_specs if s['noise_source'] == source]}")
        print(f"[WARNING] {NOISE_SOURCE_CAVEATS.get(source, CORPUS_NOISE_DISCLAIMER)}")

    gpu_names = []
    try:
        import torch
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    gate = str(cfg.get("required_gpu_substring") or "")
    if gate and (not gpu_names or gate.lower() not in gpu_names[0].lower()):
        raise RuntimeError(f"GPU gate {gate!r} not satisfied, saw {gpu_names}")

    from faster_whisper import WhisperModel
    from edge_frontend import GtcrnDenoiser, SileroVAD

    asr_dir = resolve_asr_dir(cfg)
    print(f"[paths] asr={asr_dir}")
    t0 = time.perf_counter()
    asr = WhisperModel(str(asr_dir), device=cfg["device"], device_index=0,
                       compute_type=cfg["asr_compute_type"], cpu_threads=4, num_workers=1)
    asr_load_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    denoiser = GtcrnDenoiser(num_threads=2)
    denoiser_load_ms = (time.perf_counter() - t0) * 1000

    def transcribe(audio: np.ndarray) -> tuple:
        t = time.perf_counter()
        segments, _ = asr.transcribe(np.ascontiguousarray(audio, dtype=np.float32),
                                     language="vi", beam_size=1, vad_filter=False,
                                     condition_on_previous_text=False)
        text = " ".join(seg.text for seg in segments).strip()
        return text, (time.perf_counter() - t) * 1000

    def denoise(audio: np.ndarray) -> tuple:
        t = time.perf_counter()
        out, out_sr = denoiser.denoise(np.ascontiguousarray(audio, dtype=np.float32), sr)
        ms = (time.perf_counter() - t) * 1000
        if out_sr != sr:
            n = int(round(len(out) * sr / out_sr))
            out = np.interp(np.linspace(0, len(out), n, endpoint=False),
                            np.arange(len(out)), out).astype(np.float32)
        return np.ascontiguousarray(out, dtype=np.float32), ms

    # vad_filter=False above is deliberate: faster-whisper's internal VAD would
    # silently pre-clean the noisy audio and we would be measuring that VAD, not
    # the noise. The clean-condition numbers here are therefore not directly
    # comparable to Results/kaggle_t4 (which ran with vad_filter=True).

    corpus = load_vivos(vivos_root(), int(cfg["samples"]) + int(cfg["warmup_samples"]))
    warm_n = int(cfg["warmup_samples"])
    for item in corpus[:warm_n]:
        audio = read_wav(item["path"], sr)
        transcribe(audio)
        denoise(audio)
    corpus = corpus[warm_n:]
    print(f"[corpus] {len(corpus)} scored utterances (+{warm_n} warm-up)")

    report = {
        "status": "running",
        "synthetic_noise_disclaimer": SYNTHETIC_NOISE_DISCLAIMER,
        "noise_sources_used": active_sources,
        "noise_source_caveats": {s: NOISE_SOURCE_CAVEATS.get(s, CORPUS_NOISE_DISCLAIMER)
                                 for s in active_sources},
        "noise_cells": [{k: v for k, v in s.items()} for s in noise_specs],
        "noise_corpus_manifest": _CORPUS_CONTEXT.get("manifest"),
        "noise_corpus_root": str(_CORPUS_CONTEXT.get("root", "")),
        "musan_status": ("NOT INCLUDED. MUSAN ships only as one 11 GB tarball "
                         "(openslr.org/17, CC BY 4.0) with no official subset; the "
                         "third-party 50-file slices on the HF Hub have no verifiable "
                         "provenance. No MUSAN claim may be made from this run."),
        "config": cfg,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_names": gpu_names,
            "on_kaggle": ON_KAGGLE,
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
        },
        "model_load_ms": {"asr": asr_load_ms, "gtcrn": denoiser_load_ms},
        "notes": [
            "ASR runs with vad_filter=False so the measured damage is the noise, "
            "not faster-whisper's internal VAD; clean WER here is therefore not "
            "identical to the vad_filter=True numbers in Results/kaggle_t4.",
            "SNR is referenced to the RMS of the speech-active region, not the "
            "whole file, so leading/trailing silence does not shift the level.",
            "Denoiser enhancement is reported as SI-SDR improvement measured in "
            "this run; the +10.4 dB in prototype/EDGE_UPGRADES.md is NOT reused.",
        ],
    }

    # ---------------- clean reference ----------------
    clean_rows, clean_lat = [], []
    for item in corpus:
        audio = read_wav(item["path"], sr)
        text, ms = transcribe(audio)
        clean_lat.append(ms)
        clean_rows.append({"id": item["id"], "duration_s": len(audio) / sr,
                           "hypothesis": text, **wer_cer(item["reference"], text)})
    clean_wers = [r["wer"] for r in clean_rows if r["wer"] is not None]
    clean_cers = [r["cer"] for r in clean_rows if r["cer"] is not None]
    report["clean_baseline"] = {
        "n": len(clean_rows),
        "wer_mean": statistics.fmean(clean_wers) if clean_wers else None,
        "cer_mean": statistics.fmean(clean_cers) if clean_cers else None,
        "wer_ci": bootstrap_ci(clean_wers, int(cfg["seed"])),
        "cer_ci": bootstrap_ci(clean_cers, int(cfg["seed"]) + 1),
        "wer_per_utterance": clean_wers,
        "cer_per_utterance": clean_cers,
        "asr_latency_ms": stat_block(clean_lat),
        "rows": clean_rows,
    }
    print(f"[clean] WER {fmt(report['clean_baseline']['wer_mean'])}  "
          f"CER {fmt(report['clean_baseline']['cer_mean'])}  n={len(clean_rows)}")

    # ---------------- N1 + N2: the noise x SNR grid ----------------
    grid, all_denoise_ms = [], []
    for spec in noise_specs:
        noise_type = spec["noise_type"]
        for snr_db in cfg["snr_db"]:
            cell_rows, den_ms, sisdr_in, sisdr_out, measured = [], [], [], [], []
            skipped = []
            for index, item in enumerate(corpus):
                audio = read_wav(item["path"], sr)
                # Keyed on noise_type, not cell_id, so the synthetic realizations
                # stay bit-identical to the published noise-t4-v1 run.
                seed = int(cfg["seed"]) + 1009 * index + stable_hash(noise_type) % 9973
                try:
                    noise, noise_provenance = realize_noise(spec, len(audio), sr, seed)
                    mixed = mix_at_snr(audio, noise, snr_db, sr)
                except ValueError as exc:
                    skipped.append({"id": item["id"], "reason": str(exc)})
                    continue
                reference_speech = mixed["speech"]
                noisy = mixed["mixture"]
                measured.append(mixed["measured_snr_db"])

                noisy_text, _ = transcribe(noisy)
                enhanced, ms = denoise(noisy)
                den_ms.append(ms)
                all_denoise_ms.append(ms)
                enhanced_text, _ = transcribe(enhanced)

                cell_rows.append({
                    "id": item["id"],
                    "duration_s": len(audio) / sr,
                    "noise_provenance": noise_provenance,
                    "measured_snr_db": mixed["measured_snr_db"],
                    "peak_limited": mixed["peak_limited"],
                    "noisy": {"hypothesis": noisy_text, **wer_cer(item["reference"], noisy_text)},
                    "denoised": {"hypothesis": enhanced_text,
                                 **wer_cer(item["reference"], enhanced_text)},
                    "denoise_ms": ms,
                    "denoise_rtf": ms / 1000.0 / max(len(audio) / sr, 1e-9),
                })
                sisdr_in.append(si_sdr(reference_speech, noisy))
                sisdr_out.append(si_sdr(reference_speech, enhanced))

            def agg(key: str, metric: str) -> list:
                return [r[key][metric] for r in cell_rows if r[key][metric] is not None]

            noisy_wer, den_wer = agg("noisy", "wer"), agg("denoised", "wer")
            noisy_cer, den_cer = agg("noisy", "cer"), agg("denoised", "cer")
            paired = [(a, b) for a, b in zip(noisy_wer, den_wer)]
            delta = [b - a for a, b in paired]
            cell = {
                "noise_source": spec["noise_source"],
                "cell_id": spec["cell_id"],
                "noise_type": noise_type,
                "noise_description": spec["noise_description"],
                "noise_source_caveat": spec["noise_source_caveat"],
                "requested_snr_db": snr_db,
                "measured_snr_db": stat_block(measured),
                "n": len(cell_rows),
                "skipped": skipped,
                "noisy": {
                    "wer_mean": statistics.fmean(noisy_wer) if noisy_wer else None,
                    "cer_mean": statistics.fmean(noisy_cer) if noisy_cer else None,
                    "wer_ci": bootstrap_ci(noisy_wer, int(cfg["seed"]) + 7),
                    "wer_per_utterance": noisy_wer, "cer_per_utterance": noisy_cer,
                },
                "denoised": {
                    "wer_mean": statistics.fmean(den_wer) if den_wer else None,
                    "cer_mean": statistics.fmean(den_cer) if den_cer else None,
                    "wer_ci": bootstrap_ci(den_wer, int(cfg["seed"]) + 8),
                    "wer_per_utterance": den_wer, "cer_per_utterance": den_cer,
                },
                "denoiser_wer_delta_mean": statistics.fmean(delta) if delta else None,
                "denoiser_wer_delta_ci": bootstrap_ci(delta, int(cfg["seed"]) + 9),
                # Three-way on purpose: a mean delta of exactly 0 is a tie, and
                # calling a tie "hurts" would overstate the headline finding.
                "denoiser_verdict": (None if not delta else
                                     "helps" if statistics.fmean(delta) < 0 else
                                     "hurts" if statistics.fmean(delta) > 0 else "tie"),
                "si_sdr_in_db": stat_block(sisdr_in),
                "si_sdr_out_db": stat_block(sisdr_out),
                "si_sdr_improvement_db": stat_block(
                    [b - a for a, b in zip(sisdr_in, sisdr_out)]),
                "denoise_latency_ms": stat_block(den_ms),
                "denoise_rtf": stat_block([r["denoise_rtf"] for r in cell_rows]),
                "rows": cell_rows,
            }
            grid.append(cell)
            n_wer = cell["noisy"]["wer_mean"]
            d_wer = cell["denoised"]["wer_mean"]
            verdict = str(cell["denoiser_verdict"]).upper()
            print(f"[grid] {spec['cell_id']:>18} @ {snr_db:>5} dB  "
                  f"noisy WER {fmt(n_wer)} -> denoised {fmt(d_wer)}  "
                  f"({verdict}, dWER {fmt(cell['denoiser_wer_delta_mean'], '+.4f')}, "
                  f"SI-SDRi {fmt(cell['si_sdr_improvement_db'].get('mean'), '+.2f')} dB)")

    report["experiment_n1_n2_grid"] = grid
    report["denoise_latency_ms_all_cells"] = stat_block(all_denoise_ms)

    # ---------------- N3: VAD false triggers on noise-only audio ----------------
    if not cli.no_vad:
        vad = SileroVAD()
        seg_len = int(sr * float(cfg["vad_segment_s"]))
        speech_level = statistics.fmean(
            [speech_active_rms(read_wav(item["path"], sr), sr) for item in corpus])
        vad_cells = []
        for spec in noise_specs:
            noise_type = spec["noise_type"]
            for snr_db in cfg["snr_db"]:
                # A noise-only segment carries no speech, so "SNR" is defined by
                # the absolute level the same noise would have had against the
                # corpus mean speech-active RMS at this SNR. That keeps the VAD
                # cell on the same x-axis as the WER cells.
                amplitude = speech_level / (10.0 ** (snr_db / 20.0))
                fired_raw, fired_denoised = 0, 0
                for trial in range(int(cfg["vad_trials_per_cell"])):
                    seed = int(cfg["seed"]) + 7919 * trial + stable_hash(noise_type) % 9973
                    noise, _ = realize_noise(spec, seg_len, sr, seed)
                    noise = np.clip(noise * amplitude, -0.99, 0.99).astype(np.float32)
                    if vad.is_speech(noise, sr):
                        fired_raw += 1
                    enhanced, _ = denoise(noise)
                    if vad.is_speech(enhanced, sr):
                        fired_denoised += 1
                trials = int(cfg["vad_trials_per_cell"])
                vad_cells.append({
                    "noise_source": spec["noise_source"],
                    "cell_id": spec["cell_id"],
                    "noise_type": noise_type,
                    "equivalent_snr_db": snr_db,
                    "noise_rms": float(amplitude),
                    "trials": trials,
                    "false_trigger_rate_raw": fired_raw / trials,
                    "false_trigger_rate_after_gtcrn": fired_denoised / trials,
                })
                print(f"[vad ] {spec['cell_id']:>18} @ {snr_db:>5} dB-equiv  "
                      f"false-trigger raw {fired_raw}/{trials}  "
                      f"after GTCRN {fired_denoised}/{trials}")
        report["experiment_n3_vad_false_trigger"] = {
            "definition": ("Fraction of noise-only segments (no target speech "
                           "present) on which SileroVAD.is_speech returns True. "
                           "Every such trigger costs a full ASR pass on nothing."),
            "caveat_for_corpus_cells": (
                "'False' trigger is only well defined for the synthetic cells and "
                "for recorded environments with no talkers. DEMAND PCAFETER is real "
                "multi-talker babble and DKITCHEN/PSTATION contain intermittent real "
                "speech, so a trigger there is the VAD doing its job on somebody "
                "else's voice, not an error. Read those rows as 'how often this "
                "environment opens the gate', and do not pool them with the "
                "synthetic false-trigger rate."),
            "segment_s": float(cfg["vad_segment_s"]),
            "corpus_mean_speech_active_rms": speech_level,
            "cells": vad_cells,
        }
    else:
        report["experiment_n3_vad_false_trigger"] = {"skipped": "--no-vad"}

    # ---------------- headline ----------------
    hurt = [c for c in grid if c["denoiser_verdict"] == "hurts"]
    helped = [c for c in grid if c["denoiser_verdict"] == "helps"]
    tied = [c for c in grid if c["denoiser_verdict"] == "tie"]
    deltas = [c["denoiser_wer_delta_mean"] for c in grid
              if c["denoiser_wer_delta_mean"] is not None]
    worst = max(grid, key=lambda c: c["denoiser_wer_delta_mean"] or -9e9) if grid else None
    best = min(grid, key=lambda c: c["denoiser_wer_delta_mean"] or 9e9) if grid else None
    report["headline"] = {
        "cells_total": len(grid),
        "cells_denoiser_helps": len(helped),
        "cells_denoiser_hurts": len(hurt),
        "cells_tie": len(tied),
        "mean_wer_delta_all_cells": statistics.fmean(deltas) if deltas else None,
        "worst_cell": None if worst is None else {
            "cell_id": worst["cell_id"], "noise_source": worst["noise_source"],
            "noise_type": worst["noise_type"], "snr_db": worst["requested_snr_db"],
            "wer_delta": worst["denoiser_wer_delta_mean"],
            "si_sdr_improvement_db": worst["si_sdr_improvement_db"].get("mean")},
        "best_cell": None if best is None else {
            "cell_id": best["cell_id"], "noise_source": best["noise_source"],
            "noise_type": best["noise_type"], "snr_db": best["requested_snr_db"],
            "wer_delta": best["denoiser_wer_delta_mean"],
            "si_sdr_improvement_db": best["si_sdr_improvement_db"].get("mean")},
        # Broken out per source on purpose: a tally that pooled real DEMAND
        # cells with synthetic ones would be a number nobody could interpret.
        "by_noise_source": {
            source: {
                "cells_total": len([c for c in grid if c["noise_source"] == source]),
                "cells_denoiser_helps": len([c for c in grid if c["noise_source"] == source
                                             and c["denoiser_verdict"] == "helps"]),
                "cells_denoiser_hurts": len([c for c in grid if c["noise_source"] == source
                                             and c["denoiser_verdict"] == "hurts"]),
                "mean_wer_delta": (statistics.fmean(d) if (d := [
                    c["denoiser_wer_delta_mean"] for c in grid
                    if c["noise_source"] == source
                    and c["denoiser_wer_delta_mean"] is not None]) else None),
                "caveat": NOISE_SOURCE_CAVEATS.get(source, CORPUS_NOISE_DISCLAIMER),
            } for source in active_sources},
        "denoise_latency_ms": report["denoise_latency_ms_all_cells"],
        "interpretation": (
            "A positive wer_delta means GTCRN made the transcript WORSE than the "
            "raw noisy audio. Perceptual enhancement (SI-SDR up) and ASR accuracy "
            "can move in opposite directions; if they do here, the deployment "
            "decision follows WER, not SI-SDR."),
        "caveat": {s: NOISE_SOURCE_CAVEATS.get(s, CORPUS_NOISE_DISCLAIMER)
                   for s in active_sources},
    }

    report["status"] = "complete"
    if cli.out:
        out_path = Path(cli.out)
    elif ON_KAGGLE:
        out_path = Path("/kaggle/working") / f"voiceai_{cfg['condition']}.json"
    else:
        out_path = PROJECT / "Results" / f"noise_robustness_{cfg['condition']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"clean WER {fmt(report['clean_baseline']['wer_mean'])} / "
          f"CER {fmt(report['clean_baseline']['cer_mean'])}  (n={report['clean_baseline']['n']})")
    print(f"GTCRN helps in {len(helped)}/{len(grid)} cells, hurts in {len(hurt)}/{len(grid)}, "
          f"tie in {len(tied)}/{len(grid)}")
    if deltas:
        print(f"mean WER change from GTCRN across all cells: "
              f"{fmt(statistics.fmean(deltas), '+.4f')} (positive = WORSE)")
    lat = report["denoise_latency_ms_all_cells"]
    if lat.get("n"):
        print(f"GTCRN latency p50 {lat['p50']:.1f} ms  p95 {lat['p95']:.1f} ms  "
              f"(mean {lat['mean']:.1f} ms, n={lat['n']})")
    for source in active_sources:
        block = report["headline"]["by_noise_source"][source]
        print(f"  [{source}] helps {block['cells_denoiser_helps']}/{block['cells_total']}, "
              f"hurts {block['cells_denoiser_hurts']}/{block['cells_total']}, "
              f"mean dWER {fmt(block['mean_wer_delta'], '+.4f')}")
    for source in active_sources:
        print(f"\n*** [{source}] {NOISE_SOURCE_CAVEATS.get(source, CORPUS_NOISE_DISCLAIMER)} ***")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
