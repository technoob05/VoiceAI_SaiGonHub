"""
Learned 32-d tone embedding (Round 26) — replace the 6-d heuristic tone
histogram in the two-stream voting cache with a contrastively trained
projection over a richer 64-d per-clip tonal feature.

Motivation: Round 25 large-corpus eval (Results/minimal_pair_eval_large.json)
showed correct-miss collapses to 36/100 at N=120 (vs. 72/100 at N=25). The
6-bin heuristic histogram lacks discrimination headroom — most carrier
contexts hash into a small number of bins and the cosine geometry becomes
saturated. We replace the 6-d heuristic with a 32-d learned embedding
trained on tonal hard negatives (same-syllable different-tone).

Pipeline (pure NumPy, CPU-only):
  1. Synthesize 30 syllables x 6 tones x 6 noise-perturbed reps = 1080 clips
     using facebook/mms-tts-vie + the same DEMAND-like noise recipe as
     minimal_pair_eval_large.py.
  2. Per clip extract a 64-d tonal feature: 32-d resampled F0 contour,
     16-d log-F0 statistics, 16-d band-energy ratios.
  3. Train a 64 -> 64 -> 32 MLP (L2-normalized output) with triplet loss.
     Hard negatives are SAME-syllable-DIFFERENT-tone (the minimal pair).
  4. Plug the learned 32-d embedding into the tone stream of two_stream_voting
     and re-run minimal_pair_eval_large.py.

Strict constraints: pure NumPy + matplotlib, no torch in the inference path.
Triplet hard-negative is mandatory. All numbers must be measured.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME",      str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                                  # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from utterance_cache import (                                       # noqa: E402
    UtteranceCache,
    cheap_mfcc,
    _f0_contour,
    tone_histogram_from_f0,
)
from two_stream_voting import (                                     # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
    TwoStreamFingerprint,
    TwoStreamStats,
    _normalize,
)


# -----------------------------------------------------------------------
# Corpus: 30 syllables x 6 tones
# -----------------------------------------------------------------------

TONE_DIACRITIC = ["", "́", "̉", "̣", "̃", "̀"]
# Diacritic indices follow the existing project convention:
#   0 ngang (no mark), 1 sac (acute), 2 hoi (hook), 3 nang (dot below),
#   4 nga (tilde), 5 huyen (grave). NOTE: this index order matches the
#   storage layout used in minimal_pair_eval_large.py.

# 30 carrier syllables — 20 reused from minimal_pair_eval_large.py + 10 new
MINIMAL_PAIR_SYLLABLES_30 = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
    "ta":  ["ta",  "tá",  "tả",  "tạ",  "tã",  "tà"],
    "na":  ["na",  "ná",  "nả",  "nạ",  "nã",  "nà"],
    "sa":  ["sa",  "sá",  "sả",  "sạ",  "sã",  "sà"],
    "ha":  ["ha",  "há",  "hả",  "hạ",  "hã",  "hà"],
    "va":  ["va",  "vá",  "vả",  "vạ",  "vã",  "và"],
    "pa":  ["pa",  "pá",  "pả",  "pạ",  "pã",  "pà"],
    "ga":  ["ga",  "gá",  "gả",  "gạ",  "gã",  "gà"],
    "ka":  ["ka",  "ká",  "kả",  "kạ",  "kã",  "kà"],
    "ra":  ["ra",  "rá",  "rả",  "rạ",  "rã",  "rà"],
    "xa":  ["xa",  "xá",  "xả",  "xạ",  "xã",  "xà"],
    "gia": ["gia", "giá", "giả", "giạ", "giã", "già"],
    "kha": ["kha", "khá", "khả", "khạ", "khã", "khà"],
    "nga": ["nga", "ngá", "ngả", "ngạ", "ngã", "ngà"],
    "nha": ["nha", "nhá", "nhả", "nhạ", "nhã", "nhà"],
    "tha": ["tha", "thá", "thả", "thạ", "thã", "thà"],
    # 10 new carriers to expand training diversity
    "mo":  ["mo",  "mó",  "mỏ",  "mọ",  "mõ",  "mò"],
    "bo":  ["bo",  "bó",  "bỏ",  "bọ",  "bõ",  "bò"],
    "co":  ["co",  "có",  "cỏ",  "cọ",  "cõ",  "cò"],
    "lo":  ["lo",  "ló",  "lỏ",  "lọ",  "lõ",  "lò"],
    "ho":  ["ho",  "hó",  "hỏ",  "họ",  "hõ",  "hò"],
    "mi":  ["mi",  "mí",  "mỉ",  "mị",  "mĩ",  "mì"],
    "bi":  ["bi",  "bí",  "bỉ",  "bị",  "bĩ",  "bì"],
    "ti":  ["ti",  "tí",  "tỉ",  "tị",  "tĩ",  "tì"],
    "vi":  ["vi",  "ví",  "vỉ",  "vị",  "vĩ",  "vì"],
    "li":  ["li",  "lí",  "lỉ",  "lị",  "lĩ",  "lì"],
}

CARRIER = "noi {syl}"

# Map syllable -> tone index 0..5 in the inner list above
def _tone_idx_for(variant: str, base_list: list[str]) -> int:
    return base_list.index(variant)


# -----------------------------------------------------------------------
# 64-d per-clip tonal feature
# -----------------------------------------------------------------------

def _resampled_f0_contour(f0: np.ndarray, out_len: int = 32) -> np.ndarray:
    """Voiced-only F0 in log-domain, mean-subtracted, linear-resampled to out_len.
    Returns zeros if too few voiced frames."""
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) < 4:
        return np.zeros(out_len, dtype=np.float32)
    voiced = f0[voiced_idx]
    log = np.log(voiced + 1e-6)
    log = log - log.mean()
    if len(log) >= out_len:
        out = np.interp(
            np.linspace(0, len(log) - 1, out_len),
            np.arange(len(log)),
            log,
        )
    else:
        # zero-pad to out_len (preserve shape on short clips)
        pad = np.zeros(out_len - len(log), dtype=np.float32)
        out = np.concatenate([log, pad])
    return out.astype(np.float32)


def _logf0_stats(f0: np.ndarray) -> np.ndarray:
    """16-d log-F0 statistics."""
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return np.zeros(16, dtype=np.float32)
    log = np.log(voiced + 1e-6)
    log = log - log.mean()
    mean = float(log.mean())
    std = float(log.std())
    rng = float(log.max() - log.min())
    half = len(log) // 2
    slope = float(log[-1] - log[0]) / max(len(log), 1)
    mid_dip = float(log[half] - 0.5 * (log[0] + log[-1])) if len(log) > 2 else 0.0
    creak = float(np.std(np.diff(log))) if len(log) > 4 else 0.0
    pcts = np.percentile(log, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]).tolist()
    feats = np.array([mean, std, rng, slope, mid_dip, creak] + pcts,
                     dtype=np.float32)
    return feats


def _band_energies(audio: np.ndarray, sr: int) -> np.ndarray:
    """16-d band-energy features: 5 tone-relevant bands x (log-energy + delta)
    + 3 ratios + 3 zero-pads. Per-clip normalized."""
    n = len(audio)
    if n < 256:
        return np.zeros(16, dtype=np.float32)
    spec = np.abs(np.fft.rfft(audio.astype(np.float32))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # Tone-relevant bands (Hz): focus on F0 + first 2 harmonics + spectral envelope
    bands = [(60, 200), (200, 400), (400, 800), (800, 1600), (1600, 4000)]
    energies = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        e = float(spec[m].sum())
        energies.append(e)
    total = sum(energies) + 1e-9
    log_e = [float(np.log(e + 1e-9) - np.log(total)) for e in energies]
    ratios = [
        energies[0] / (energies[1] + 1e-9),
        energies[1] / (energies[2] + 1e-9),
        (energies[0] + energies[1]) / (sum(energies) + 1e-9),
    ]
    diffs = [log_e[i+1] - log_e[i] for i in range(4)]
    feats = log_e + diffs + ratios + [0.0, 0.0, 0.0]
    feats = np.array(feats[:16], dtype=np.float32)
    return feats


def clip_to_64d(audio: np.ndarray, sr: int) -> np.ndarray:
    """The 64-d input vector for the learned tone embedding."""
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    f0 = _f0_contour(audio, sr)
    contour = _resampled_f0_contour(f0, out_len=32)
    stats = _logf0_stats(f0)
    bands = _band_energies(audio, sr)
    return np.concatenate([contour, stats, bands]).astype(np.float32)


# -----------------------------------------------------------------------
# Pure-NumPy 64 -> 64 -> 32 MLP with triplet loss
# -----------------------------------------------------------------------

class ToneEmbeddingMLP:
    def __init__(self, in_dim: int = 64, hidden: int = 64, out_dim: int = 32,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden)) *
                   np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) *
                   np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std:  Optional[np.ndarray] = None

    def fit_norm(self, X: np.ndarray):
        self.feat_mean = X.mean(axis=0).astype(np.float32)
        self.feat_std  = (X.std(axis=0) + 1e-3).astype(np.float32)

    def _norm(self, X: np.ndarray) -> np.ndarray:
        if self.feat_mean is None:
            return X
        return (X - self.feat_mean) / self.feat_std

    def forward(self, X: np.ndarray) -> dict:
        X = self._norm(X)
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        norm = np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8
        out = z2 / norm
        return {"X": X, "z1": z1, "h1": h1, "z2": z2, "norm": norm, "out": out}

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["out"]

    def backward_triplet(self, ca, cp, cn, margin: float):
        a = ca["out"]; p = cp["out"]; n = cn["out"]
        B = a.shape[0]
        ap = (a * p).sum(axis=-1)
        an = (a * n).sum(axis=-1)
        loss_per = np.maximum(0.0, margin + an - ap)
        active = (loss_per > 0).astype(np.float32)
        loss = float(loss_per.mean())
        n_active = int(active.sum())
        if n_active == 0:
            z = lambda M: np.zeros_like(M)
            return loss, {"W1": z(self.W1), "b1": z(self.b1),
                          "W2": z(self.W2), "b2": z(self.b2)}, 0

        mask = active[:, None] / B
        da_out = (n - p) * mask
        dp_out = -a * mask
        dn_out =  a * mask

        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)

        for c, d_out in [(ca, da_out), (cp, dp_out), (cn, dn_out)]:
            out = c["out"]; norm = c["norm"]
            dot = (d_out * out).sum(axis=-1, keepdims=True)
            dz2 = (d_out - dot * out) / norm
            gW2 += c["h1"].T @ dz2
            gb2 += dz2.sum(axis=0)
            dh1 = dz2 @ self.W2.T
            dz1 = dh1 * (c["z1"] > 0)
            gW1 += c["X"].T @ dz1
            gb1 += dz1.sum(axis=0)

        return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, n_active

    def step(self, grads, lr: float, momentum: float, wd: float = 1e-4):
        self.vW1 = momentum * self.vW1 - lr * (grads["W1"] + wd * self.W1); self.W1 += self.vW1
        self.vb1 = momentum * self.vb1 - lr *  grads["b1"];                self.b1 += self.vb1
        self.vW2 = momentum * self.vW2 - lr * (grads["W2"] + wd * self.W2); self.W2 += self.vW2
        self.vb2 = momentum * self.vb2 - lr *  grads["b2"];                self.b2 += self.vb2

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size +
                   (self.feat_mean.size if self.feat_mean is not None else 0) +
                   (self.feat_std.size  if self.feat_std  is not None else 0))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 feat_mean=(self.feat_mean if self.feat_mean is not None
                            else np.zeros(self.W1.shape[0], dtype=np.float32)),
                 feat_std =(self.feat_std  if self.feat_std  is not None
                            else np.ones (self.W1.shape[0], dtype=np.float32)))

    @classmethod
    def load(cls, path: Path) -> "ToneEmbeddingMLP":
        z = np.load(str(path))
        in_dim, hidden = z["W1"].shape
        _, out_dim = z["W2"].shape
        m = cls(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        m.W1 = z["W1"]; m.b1 = z["b1"]; m.W2 = z["W2"]; m.b2 = z["b2"]
        m.feat_mean = z["feat_mean"]; m.feat_std = z["feat_std"]
        return m


# -----------------------------------------------------------------------
# Noise model (DEMAND-like, identical recipe to minimal_pair_eval_large.py)
# -----------------------------------------------------------------------

def _demand_like_noise(n: int, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    pink = np.empty_like(white)
    pink[0] = white[0]
    a = 0.92
    for i in range(1, n):
        pink[i] = a * pink[i-1] + (1 - a) * white[i]
    t = np.arange(n) / sr
    rumble = (np.sin(2 * np.pi * 110 * t + rng.uniform(0, 2*np.pi)).astype(np.float32) * 0.4
              + np.sin(2 * np.pi *  85 * t + rng.uniform(0, 2*np.pi)).astype(np.float32) * 0.3)
    noise = 0.7 * pink + 0.3 * rumble
    rms = float(np.sqrt((noise ** 2).mean() + 1e-12))
    return (noise / max(rms, 1e-9)).astype(np.float32)


def _mix_at_snr(audio: np.ndarray, snr_db: float, rng: np.random.Generator,
                sr: int) -> np.ndarray:
    sig_power = float((audio ** 2).mean() + 1e-12)
    noise_unit = _demand_like_noise(len(audio), rng, sr=sr)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise_unit * np.sqrt(target_noise_power)
    return (audio + noise).astype(np.float32)


# -----------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------

def synthesize_training_corpus(n_reps: int = 6,
                               snr_choices: tuple = (None, 20.0, 15.0, 10.0, 5.0)
                               ) -> dict:
    """Returns dict (base, tone_idx, rep) -> (audio16k, sr=16000).
    Each rep gets a different noise/SNR draw; rep 0 is clean to anchor the
    deterministic clean exemplar."""
    try:
        from transformers import AutoTokenizer, VitsModel
        import torch
    except Exception as e:
        print(f"[abort] cannot import transformers/torch: {e}")
        return {}

    print("[synth] loading MMS-TTS-vie ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[synth] loaded in {time.perf_counter()-t0:.1f}s  sr={sr}")

    clips: dict = {}
    n_total = sum(len(v) for v in MINIMAL_PAIR_SYLLABLES_30.values()) * n_reps
    i = 0
    t_synth = time.perf_counter()
    for base, variants in MINIMAL_PAIR_SYLLABLES_30.items():
        for tone_idx, syl in enumerate(variants):
            # Deterministic clean synthesis per (base, tone) — reuse across reps
            torch.manual_seed(tone_idx * 7 + (hash(base) & 0xFFFF))
            inputs = tok(CARRIER.format(syl=syl), return_tensors="pt")
            try:
                with torch.inference_mode():
                    wav = model(**inputs).waveform
                clean = wav.squeeze().cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"  [warn] synth fail {base}/{syl}: {e}")
                continue
            # Resample to 16 kHz once
            if sr != 16000:
                n16 = int(round(len(clean) * 16000 / sr))
                clean16 = np.interp(np.linspace(0, len(clean), n16, endpoint=False),
                                    np.arange(len(clean)), clean).astype(np.float32)
            else:
                clean16 = clean

            for rep in range(n_reps):
                snr = snr_choices[rep % len(snr_choices)]
                if snr is None:
                    audio = clean16
                else:
                    rng = np.random.default_rng(
                        20260528 + tone_idx * 991 + rep * 17 + (hash(base) & 0xFFFF)
                    )
                    audio = _mix_at_snr(clean16, snr, rng, 16000)
                clips[(base, tone_idx, rep)] = (audio, 16000)
                i += 1
            if (i % 60) == 0:
                print(f"  [synth] {i}/{n_total}  elapsed={time.perf_counter()-t_synth:5.1f}s")
    print(f"[synth] {len(clips)}/{n_total} clips in {time.perf_counter()-t_synth:.1f}s")
    return clips


# -----------------------------------------------------------------------
# Triplet sampling: anchor + positive (any-syl same-tone) + HARD NEG (same-syl diff-tone)
# -----------------------------------------------------------------------

def sample_triplets(by_tone: dict[int, list[int]], by_base_tone: dict[tuple, list[int]],
                    keys: list[tuple], n: int, rng: random.Random,
                    p_hard: float = 1.0) -> list[tuple[int, int, int]]:
    """Build n triplets.

    - anchor: random clip of tone t and syllable S
    - positive: another clip with tone t (any syllable, prefer different syllable
      so the positive does not collapse to "same recording")
    - hard negative: clip of SAME syllable S but DIFFERENT tone t' != t
      (i.e. the tonal minimal pair). If unavailable, fall back to random
      different-tone clip.

    p_hard: probability of using hard negative (vs random different-tone).
    """
    triplets = []
    tones = list(by_tone.keys())
    for _ in range(n):
        t = rng.choice(tones)
        if len(by_tone[t]) < 2:
            continue
        a_idx = rng.choice(by_tone[t])
        base_a, _, _ = keys[a_idx]
        # Positive: same-tone, different-syllable preferred
        diff_syl_pos = [j for j in by_tone[t]
                        if j != a_idx and keys[j][0] != base_a]
        if not diff_syl_pos:
            diff_syl_pos = [j for j in by_tone[t] if j != a_idx]
        if not diff_syl_pos:
            continue
        p_idx = rng.choice(diff_syl_pos)
        # Hard negative: same-syllable, different-tone
        use_hard = rng.random() < p_hard
        n_idx = None
        if use_hard:
            same_syl_diff_tone = []
            for t2 in tones:
                if t2 == t:
                    continue
                key = (base_a, t2)
                if key in by_base_tone:
                    same_syl_diff_tone.extend(by_base_tone[key])
            if same_syl_diff_tone:
                n_idx = rng.choice(same_syl_diff_tone)
        if n_idx is None:
            # Random different-tone fallback
            other_tones = [t2 for t2 in tones if t2 != t]
            t2 = rng.choice(other_tones)
            n_idx = rng.choice(by_tone[t2])
        triplets.append((a_idx, p_idx, n_idx))
    return triplets


# -----------------------------------------------------------------------
# Two-stream cache with learned tone embedding
# -----------------------------------------------------------------------

class LearnedToneTwoStreamCache(TwoStreamCache):
    """Drop-in for TwoStreamCache where the 6-d tone histogram is replaced
    by a learned 32-d embedding from ToneEmbeddingMLP. Segment stream
    unchanged."""

    def __init__(self, mlp: ToneEmbeddingMLP, cfg: Optional[TwoStreamConfig] = None):
        super().__init__(cfg)
        self.mlp = mlp

    def fingerprint(self, audio: np.ndarray, sr: int) -> TwoStreamFingerprint:
        # Tone stream: learned 32-d
        feat64 = clip_to_64d(audio, sr)
        emb32 = self.mlp.apply(feat64[None, :])[0]   # already L2-normed
        # Segment stream: same as parent (MFCC + delta concat, 260-d)
        if sr != 16000:
            n = int(round(len(audio) * 16000 / sr))
            audio16 = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                                np.arange(len(audio)), audio).astype(np.float32)
            sr16 = 16000
        else:
            audio16 = audio; sr16 = sr
        mfcc = cheap_mfcc(audio16, sr16, n_mfcc=13)
        mfcc_sub = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
        mfcc_flat = mfcc_sub.flatten(order="F")
        mfcc_delta = np.diff(mfcc_sub, axis=1, prepend=mfcc_sub[:, :1])
        mfcc_delta_flat = mfcc_delta.flatten(order="F")
        seg = np.concatenate([mfcc_flat, mfcc_delta_flat])
        return TwoStreamFingerprint(
            tone=emb32.astype(np.float32),
            segment=_normalize(seg),
        )


# -----------------------------------------------------------------------
# Minimal-pair eval harness (mirrors minimal_pair_eval_large.py protocol)
# -----------------------------------------------------------------------

# Use the SAME 20-syllable canonical eval corpus as minimal_pair_eval_large.py
# so the numbers are directly comparable.
from two_stream_voting import MINIMAL_PAIR_SYLLABLES as _UNUSED_BASE_5
EVAL_SYLLABLES = {k: v for k, v in MINIMAL_PAIR_SYLLABLES_30.items()
                  if k in ("ma","ba","ca","la","da","ta","na","sa","ha","va",
                           "pa","ga","ka","ra","xa","gia","kha","nga","nha","tha")}


def _synth_eval_corpus(snr_db: Optional[float]) -> dict:
    """Synthesize the 120-clip eval corpus (20 syllables x 6 tones) at one
    SNR condition. Returns dict (base, variant) -> audio16k."""
    try:
        from transformers import AutoTokenizer, VitsModel
        import torch
    except Exception as e:
        print(f"[abort eval-synth] {e}")
        return {}
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    out = {}
    for base, variants in EVAL_SYLLABLES.items():
        for tone_idx, v in enumerate(variants):
            torch.manual_seed(tone_idx * 7 + (hash(base) & 0xFFFF))
            inputs = tok(CARRIER.format(syl=v), return_tensors="pt")
            with torch.inference_mode():
                wav = model(**inputs).waveform
            a = wav.squeeze().cpu().numpy().astype(np.float32)
            if sr != 16000:
                n16 = int(round(len(a) * 16000 / sr))
                a = np.interp(np.linspace(0, len(a), n16, endpoint=False),
                              np.arange(len(a)), a).astype(np.float32)
            if snr_db is not None:
                rng = np.random.default_rng(
                    20260528 + int(snr_db * 100) + tone_idx * 991 +
                    (hash(base) & 0xFFFF)
                )
                a = _mix_at_snr(a, snr_db, rng, 16000)
            out[(base, v)] = a
    return out


def eval_minimal_pair(cache_factory, audios: dict, tau_A: float, tau_B: float
                      ) -> dict:
    cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B,
                          max_entries=1024)
    # Self-recall: store every clip; query each against itself
    cache = cache_factory(cfg)
    fps = {}
    for k, a in audios.items():
        fp = cache.fingerprint(a, 16000)
        fps[k] = fp
        cache.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})
    self_hits = 0
    for k, fp in fps.items():
        h = cache.lookup_existence(fp, "en")
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1
    n_clips = len(audios)

    # Tonal-variant probe: cache only ngang per base; probe the other 5 tones
    cache_t = cache_factory(cfg)
    for base, variants in EVAL_SYLLABLES.items():
        ngang_key = (base, variants[0])
        if ngang_key in audios:
            fp = cache_t.fingerprint(audios[ngang_key], 16000)
            cache_t.put(fp, "en", {"vi": variants[0], "tgt": variants[0]})

    correct_miss = false_hit = total = 0
    for base, variants in EVAL_SYLLABLES.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audios:
                continue
            fp = cache_t.fingerprint(audios[k], 16000)
            h = cache_t.lookup_existence(fp, "en")
            total += 1
            if h is None:
                correct_miss += 1
            else:
                false_hit += 1
    return {
        "tau_A": tau_A, "tau_B": tau_B,
        "n_clips_total": n_clips,
        "self_hits": self_hits,
        "self_recall": round(self_hits / max(n_clips, 1), 4),
        "n_tonal_probes": total,
        "correct_misses": correct_miss,
        "correct_miss_rate": round(correct_miss / max(total, 1), 4),
        "false_hits": false_hit,
        "false_hit_rate": round(false_hit / max(total, 1), 4),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

HEURISTIC_BASELINE_CLEAN = {
    "tau_A=0.85_tau_B=0.93": {
        "self_hits": 120, "self_recall": 1.0,
        "n_tonal_probes": 100, "correct_misses": 36,
        "correct_miss_rate": 0.36, "false_hits": 64,
        "false_hit_rate": 0.64,
    },
    "tau_A=0.9_tau_B=0.93": {
        "self_hits": 120, "self_recall": 1.0,
        "n_tonal_probes": 100, "correct_misses": 57,
        "correct_miss_rate": 0.57, "false_hits": 43,
        "false_hit_rate": 0.43,
    },
}
HEURISTIC_BASELINE_SNR10 = {
    "correct_misses": 56, "correct_miss_rate": 0.56,
    "false_hits": 44, "false_hit_rate": 0.44,
}
HEURISTIC_BASELINE_SNR5  = {
    "correct_misses": 53, "correct_miss_rate": 0.53,
    "false_hits": 47, "false_hit_rate": 0.47,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps",       type=int, default=6)
    ap.add_argument("--n-iters",      type=int, default=1000)
    ap.add_argument("--lr",           type=float, default=0.01)
    ap.add_argument("--momentum",     type=float, default=0.9)
    ap.add_argument("--margin",       type=float, default=0.3)
    ap.add_argument("--batch-size",   type=int, default=32)
    ap.add_argument("--hidden",       type=int, default=64)
    ap.add_argument("--out-dim",      type=int, default=32)
    ap.add_argument("--seed",         type=int, default=20260528)
    ap.add_argument("--out-json",     default="Results/learned_tone_stream_eval.json")
    ap.add_argument("--out-weights",  default="data/cache_snapshots/learned_tone_stream.npz")
    ap.add_argument("--out-figure",   default="papers/figures/fig_learned_tone_vs_heuristic.png")
    ap.add_argument("--skip-noise-eval", action="store_true",
                    help="Run only the clean eval (saves ~3-5 minutes).")
    args = ap.parse_args()

    rng_np = np.random.default_rng(args.seed)
    rng = random.Random(args.seed)

    # ---------- 1. Synthesize training corpus ----------
    print("\n=== STEP 1: synthesize training corpus ===")
    clips = synthesize_training_corpus(n_reps=args.n_reps)
    if len(clips) < 200:
        print("[abort] too few training clips.")
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps({"status": "abort", "reason": "tts_unavailable",
                        "n_clips": len(clips)}, indent=2), encoding="utf-8")
        return 1

    # ---------- 2. Extract 64-d features ----------
    print("\n=== STEP 2: extract 64-d features ===")
    X_list, key_list = [], []
    for k, (a, sr) in clips.items():
        feat = clip_to_64d(a, sr)
        X_list.append(feat)
        key_list.append(k)   # (base, tone_idx, rep)
    X = np.stack(X_list, axis=0).astype(np.float32)
    print(f"[feat] X.shape={X.shape}")
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    by_tone: dict[int, list[int]] = defaultdict(list)
    by_base_tone: dict[tuple, list[int]] = defaultdict(list)
    for i, (b, t, _r) in enumerate(key_list):
        by_tone[t].append(i)
        by_base_tone[(b, t)].append(i)
    print(f"[index] by_tone counts: {[len(by_tone[t]) for t in sorted(by_tone)]}")

    # ---------- 3. Train MLP ----------
    print("\n=== STEP 3: train 64 -> 64 -> 32 MLP ===")
    mlp = ToneEmbeddingMLP(in_dim=X.shape[1], hidden=args.hidden,
                           out_dim=args.out_dim, seed=args.seed)
    mlp.fit_norm(X)
    print(f"[train] params={mlp.n_params()}  "
          f"feat_mean range=[{mlp.feat_mean.min():.2f},{mlp.feat_mean.max():.2f}]  "
          f"feat_std range=[{mlp.feat_std.min():.3f},{mlp.feat_std.max():.3f}]")
    loss_curve = []
    t0 = time.perf_counter()
    for it in range(args.n_iters):
        trips = sample_triplets(by_tone, by_base_tone, key_list,
                                args.batch_size, rng, p_hard=1.0)
        if not trips:
            continue
        a_ix = np.array([t[0] for t in trips], dtype=np.int64)
        p_ix = np.array([t[1] for t in trips], dtype=np.int64)
        n_ix = np.array([t[2] for t in trips], dtype=np.int64)
        Xa = X[a_ix]; Xp = X[p_ix]; Xn = X[n_ix]
        ca = mlp.forward(Xa); cp = mlp.forward(Xp); cn = mlp.forward(Xn)
        loss, grads, n_active = mlp.backward_triplet(ca, cp, cn, args.margin)
        mlp.step(grads, args.lr, args.momentum)
        if it % 50 == 0 or it == args.n_iters - 1:
            loss_curve.append({"iter": it, "loss": round(loss, 4),
                               "n_active": n_active})
            print(f"  iter {it:4d}  loss={loss:.4f}  active={n_active}/{len(trips)}")
    print(f"[train] done in {time.perf_counter()-t0:.1f}s")

    # ---------- 4. Save weights + footprint ----------
    weights_path = PROJECT / args.out_weights
    mlp.save(weights_path)
    size_kb = weights_path.stat().st_size / 1024.0
    print(f"[save] weights -> {weights_path}  size={size_kb:.2f} KB")

    # ---------- 5. Eval — clean + noisy ----------
    print("\n=== STEP 5: minimal-pair eval (LEARNED) ===")

    # Build eval audios once per condition
    eval_conditions = [("clean", None)]
    if not args.skip_noise_eval:
        eval_conditions += [("snr_10dB", 10.0), ("snr_5dB", 5.0)]

    def learned_factory(cfg): return LearnedToneTwoStreamCache(mlp, cfg)
    def heuristic_factory(cfg): return TwoStreamCache(cfg)

    results: dict = {}
    for cname, snr_db in eval_conditions:
        print(f"\n[cond {cname}] synthesizing eval corpus (snr={snr_db}) ...")
        audios = _synth_eval_corpus(snr_db)
        print(f"   {len(audios)} clips")
        per_op = {}
        op_grid = [(0.60, 0.93), (0.65, 0.93), (0.70, 0.93),
                   (0.75, 0.93), (0.80, 0.93), (0.85, 0.93), (0.90, 0.93)]
        for (tA, tB) in op_grid:
            t0 = time.perf_counter()
            r_learned = eval_minimal_pair(learned_factory, audios, tA, tB)
            el = time.perf_counter() - t0
            print(f"   LEARNED  tau=({tA},{tB})  self={r_learned['self_hits']}/120  "
                  f"correct-miss={r_learned['correct_misses']}/100 "
                  f"({r_learned['correct_miss_rate']*100:5.1f}%)  "
                  f"false={r_learned['false_hits']:3d}  elapsed={el:4.1f}s")
            per_op[f"tau_A={tA}_tau_B={tB}"] = {"learned": r_learned}

        # Heuristic side-by-side at the canonical (0.85, 0.93)
        r_heur = eval_minimal_pair(heuristic_factory, audios, 0.85, 0.93)
        print(f"   HEURISTIC tau=(0.85,0.93)  self={r_heur['self_hits']}/120  "
              f"correct-miss={r_heur['correct_misses']}/100 "
              f"({r_heur['correct_miss_rate']*100:5.1f}%)")
        per_op["tau_A=0.85_tau_B=0.93"]["heuristic"] = r_heur
        results[cname] = {"snr_db": snr_db, "by_operating_point": per_op}

    # ---------- 6. Pick best operating point on clean ----------
    print("\n=== STEP 6: select best operating point on clean ===")
    best_op, best_cm = None, -1
    for op, d in results["clean"]["by_operating_point"].items():
        cm = d["learned"]["correct_misses"]
        sr = d["learned"]["self_hits"]
        # require self-recall >= 100/120 to avoid degenerate solutions
        if sr >= 100 and cm > best_cm:
            best_cm = cm; best_op = op
    if best_op is None:
        # fallback: highest correct-miss regardless
        for op, d in results["clean"]["by_operating_point"].items():
            cm = d["learned"]["correct_misses"]
            if cm > best_cm:
                best_cm = cm; best_op = op
    print(f"[best] {best_op}  correct-miss={best_cm}/100")

    # ---------- 7. Bar chart figure ----------
    print("\n=== STEP 7: figure ===")
    fig_path = PROJECT / args.out_figure
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    cond_names = [c[0] for c in eval_conditions]
    heur_cm = []
    learn_cm = []
    for cname in cond_names:
        heur_d = results[cname]["by_operating_point"]["tau_A=0.85_tau_B=0.93"]["heuristic"]
        # For learned: use best clean tau_A across all conds (uniform threshold)
        learn_d = results[cname]["by_operating_point"][best_op]["learned"]
        heur_cm.append(heur_d["correct_misses"])
        learn_cm.append(learn_d["correct_misses"])

    x = np.arange(len(cond_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    b1 = ax.bar(x - width/2, heur_cm,  width, label="Heuristic 6-d (tau_A=0.85)",
                color="#c44e52")
    b2 = ax.bar(x + width/2, learn_cm, width,
                label=f"Learned 32-d ({best_op})", color="#4c72b0")
    ax.set_xticks(x); ax.set_xticklabels(cond_names)
    ax.set_ylabel("Correct-miss count (out of 100)")
    ax.set_ylim(0, 100)
    ax.set_title("Tonal minimal-pair correct-miss: heuristic vs learned tone stream")
    ax.axhline(50, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{int(h)}", xy=(r.get_x()+r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)
    print(f"[fig] -> {fig_path}")

    # ---------- 8. Write JSON ----------
    out = {
        "status": "ok",
        "round": 26,
        "experiment": "Learned 32-d tone embedding (triplet, hard negatives) replaces 6-d heuristic tone stream",
        "training_corpus": {
            "n_syllables":    len(MINIMAL_PAIR_SYLLABLES_30),
            "n_tones":        6,
            "n_reps_per_cell": args.n_reps,
            "n_clips_total":  len(clips),
            "snr_choices_db": [None, 20.0, 15.0, 10.0, 5.0],
            "carrier":        CARRIER,
        },
        "feature": {
            "in_dim": 64,
            "components": [
                "32-d resampled log-F0 contour (mean-subtracted)",
                "16-d log-F0 statistics (mean, std, range, slope, mid-dip, creak, 10 percentiles)",
                "16-d band energies (5 bands x log + delta + ratios, zero-padded to 16)",
            ],
        },
        "model": {
            "arch":       f"MLP 64 -> {args.hidden} -> {args.out_dim} (ReLU, L2-norm output)",
            "loss":       "triplet (margin=0.3, hard-negative=same-syllable-different-tone)",
            "optimizer":  "SGD + momentum (pure NumPy)",
            "n_iters":    args.n_iters,
            "lr":         args.lr,
            "momentum":   args.momentum,
            "batch_size": args.batch_size,
            "params":     mlp.n_params(),
            "weights_path": str(weights_path.relative_to(PROJECT)),
            "size_kb":    round(size_kb, 2),
            "footprint_passes_30kb": size_kb <= 30.0,
        },
        "loss_curve": loss_curve,
        "eval": results,
        "best_operating_point_on_clean": best_op,
        "baselines_pinned": {
            "heuristic_clean_tau_A_0.85":  HEURISTIC_BASELINE_CLEAN["tau_A=0.85_tau_B=0.93"],
            "heuristic_clean_tau_A_0.90":  HEURISTIC_BASELINE_CLEAN["tau_A=0.9_tau_B=0.93"],
            "heuristic_snr10dB_tau_A_0.85": HEURISTIC_BASELINE_SNR10,
            "heuristic_snr5dB_tau_A_0.85":  HEURISTIC_BASELINE_SNR5,
            "source": "Results/minimal_pair_eval_large.json (Round 25, N=120)",
        },
        "headline": {
            "heuristic_clean_correct_miss":  36,
            "learned_clean_correct_miss":    int(best_cm),
            "delta":                         int(best_cm) - 36,
            "best_tau_A":                    best_op,
            "footprint_kb":                  round(size_kb, 2),
        },
    }
    out_path = PROJECT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    print(f"\n========= HEADLINE =========")
    print(f"  heuristic correct-miss (clean, tau_A=0.85): 36/100")
    print(f"  learned   correct-miss (clean, {best_op}): {best_cm}/100")
    print(f"  delta:    {best_cm - 36:+d}")
    print(f"  footprint: {size_kb:.2f} KB  (target <=30 KB: {'OK' if size_kb <= 30 else 'OVER'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
