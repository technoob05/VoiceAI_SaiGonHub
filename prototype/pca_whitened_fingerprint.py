"""
PCA-whitened fingerprint — data-driven weighting as alternative to manual α-tuning.

The weighted_fingerprint.py sweep (paper §6.5) failed: no manual scalar
weight on the tone component restored minimal-pair protection above TTS
noise. A learned-from-data weighting might do better:
  - Compute the 272-d raw fingerprint on a Vi training set
  - Estimate covariance matrix Σ
  - Whiten: vec' = Σ^{-1/2} (vec - μ)
  - This rescales each direction by its inverse variance, so the
    low-variance tone histogram gets boosted relative to high-variance
    MFCC trajectory

Then cosine in whitened space = Mahalanobis distance. Components that
were under-represented (tone) gain influence; components that varied
strongly across speakers (energy, MFCC absolute level) lose influence.

This is paper §3.5 alternative architecture (vs two-stream voting from
two_stream_voting.py). We test it on the same minimal-pair corpus and
compare protect-gap.

Output: Results/pca_whitened_eval.json with protect-gap before/after
whitening.

Note: this is a quick test, NOT a production-quality whitening. Real
deployment would train Σ on a tonal-minimal-pair corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME",      str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                            # noqa: E402
import soundfile as sf                                         # noqa: E402

from utterance_cache import (                                  # noqa: E402
    UtteranceCache,
    cheap_mfcc,
    tone_histogram_from_f0,
    _f0_contour,
)


def _raw_fingerprint(audio, sr):
    """Re-compute the 272-d raw concat without final L2-norm (we want pre-norm space for PCA)."""
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    f0 = _f0_contour(audio, sr)
    tone_hist = tone_histogram_from_f0(f0)
    voiced = f0[f0 > 0]
    if len(voiced) >= 4:
        log_v = np.log(voiced + 1e-6)
        f0_stats = np.array([
            log_v.mean(),
            log_v.std(),
            log_v.max() - log_v.min(),
            float(np.polyfit(np.arange(len(log_v)), log_v, 1)[0]),
        ], dtype=np.float32)
    else:
        f0_stats = np.zeros(4, dtype=np.float32)
    mfcc = cheap_mfcc(audio, sr, n_mfcc=13)
    mfcc_sub = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
    mfcc_flat = mfcc_sub.flatten(order="F")
    mfcc_delta = np.diff(mfcc_sub, axis=1, prepend=mfcc_sub[:, :1])
    mfcc_delta_flat = mfcc_delta.flatten(order="F")
    rms_all = float(np.sqrt((audio ** 2).mean() + 1e-12))
    e_bounds = np.linspace(0, len(audio), 9, dtype=int)
    chunk_rms = [float(np.sqrt((audio[e_bounds[i]:e_bounds[i+1]] ** 2).mean() + 1e-12))
                 for i in range(8)]
    energy_stats = np.array([rms_all, max(chunk_rms)], dtype=np.float32)
    return np.concatenate([tone_hist, mfcc_flat, mfcc_delta_flat, f0_stats, energy_stats]).astype(np.float32)


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def fit_whitening(fps: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean + inverse-square-root of covariance for ZCA whitening."""
    mu = fps.mean(axis=0)
    centered = fps - mu
    # Per-dim variance whitening (cheap, no full SVD).
    var = centered.var(axis=0) + eps
    inv_std = 1.0 / np.sqrt(var)
    return mu, inv_std.astype(np.float32)


def apply_whitening(fp: np.ndarray, mu: np.ndarray, inv_std: np.ndarray) -> np.ndarray:
    return _l2(((fp - mu) * inv_std).astype(np.float32))


MINIMAL_PAIR_SYLLABLES = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
}
CARRIER = "nói {syl}"


def _synth(text, tok, model):
    import torch
    inputs = tok(text, return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform
    return wav.squeeze().cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30, help="VIVOS samples to fit whitening")
    ap.add_argument("--out",     default="Results/pca_whitened_eval.json")
    args = ap.parse_args()

    print("[load] MMS-TTS-vie for minimal-pair synthesis ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    # ---- Fit whitening on real VIVOS train audio ----
    print(f"[fit] computing whitening on {args.n_train} VIVOS train utterances ...")
    train_root = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train" / "waves"
    wavs = []
    if train_root.exists():
        for spk_dir in sorted(train_root.iterdir())[:30]:
            if spk_dir.is_dir():
                for w in sorted(spk_dir.glob("*.wav"))[:1]:
                    wavs.append(w)
                    if len(wavs) >= args.n_train: break
            if len(wavs) >= args.n_train: break

    train_fps = []
    for w in wavs:
        try:
            audio, sr_w = sf.read(str(w), dtype="float32")
            if audio.ndim > 1: audio = audio.mean(axis=1)
            train_fps.append(_raw_fingerprint(audio, sr_w))
        except Exception as e:
            print(f"  [warn] failed on {w.name}: {e}")
    train_fps = np.stack(train_fps, axis=0)
    print(f"[fit] training matrix shape {train_fps.shape}")
    mu, inv_std = fit_whitening(train_fps)
    print(f"[fit] whitening fitted; inv_std min={inv_std.min():.4f} max={inv_std.max():.4f}")

    # ---- Synthesize minimal pairs ----
    print("\n[synth] generating minimal-pair clips ...")
    audio_by_key = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] {v}: {e}")
    print(f"[synth] {len(audio_by_key)} clips")

    raw_by_key = {k: _raw_fingerprint(a, sr) for k, a in audio_by_key.items()}
    whitened_by_key = {k: apply_whitening(r, mu, inv_std) for k, r in raw_by_key.items()}
    naive_l2_by_key = {k: _l2(r) for k, r in raw_by_key.items()}

    # ---- Pairwise cosines: intra-tone vs inter-syllable, naive L2 vs whitened ----
    print(f"\n{'variant':<14}{'intra_tone_cos':>17}{'inter_syl_cos':>16}{'protect_gap':>14}{'verdict':>9}")
    print("-" * 80)

    results = {}
    for label, fp_by_key in [("naive_l2", naive_l2_by_key), ("pca_whitened", whitened_by_key)]:
        intra = []
        for base, variants in MINIMAL_PAIR_SYLLABLES.items():
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    k1, k2 = (base, variants[i]), (base, variants[j])
                    if k1 in fp_by_key and k2 in fp_by_key:
                        intra.append(float(np.dot(fp_by_key[k1], fp_by_key[k2])))

        inter = []
        bases = list(MINIMAL_PAIR_SYLLABLES.keys())
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                for v1 in MINIMAL_PAIR_SYLLABLES[bases[i]]:
                    for v2 in MINIMAL_PAIR_SYLLABLES[bases[j]]:
                        k1, k2 = (bases[i], v1), (bases[j], v2)
                        if k1 in fp_by_key and k2 in fp_by_key:
                            inter.append(float(np.dot(fp_by_key[k1], fp_by_key[k2])))

        intra_m = float(np.mean(intra)) if intra else 0.0
        inter_m = float(np.mean(inter)) if inter else 0.0
        gap = inter_m - intra_m
        verdict = "✓" if gap > 0 else "✗"
        results[label] = {"intra_tone_cos": round(intra_m, 4),
                          "inter_syl_cos": round(inter_m, 4),
                          "protect_gap":   round(gap, 4)}
        print(f"  {label:<12}{intra_m:>17.4f}{inter_m:>16.4f}{gap:>+14.4f}{verdict:>9}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_train_samples": int(train_fps.shape[0]),
        "fingerprint_dim": int(train_fps.shape[1]),
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
