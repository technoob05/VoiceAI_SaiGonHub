"""
Weighted-concatenation fingerprint — fix for the §2.6 tone-drowning finding.

The naive design concatenated 5 components and L2-normalized the whole vector.
Result (`Results/minimal_pair_eval.json`):
  tone_only (6d)  → intra-tone 0.69 < inter-syl 0.72  ✓ tone HAS signal
  mfcc_only(130d) → intra-tone 0.95 > inter-syl 0.94  ✗ MFCC confuses tones
  full     (272d) → intra-tone 0.92 > inter-syl 0.91  ✗ MFCC dominates

In the full L2-normed concat, each component's contribution scales with its
component's *L2 norm*, NOT with its design importance. Tone histogram is a
6-d probability distribution (||t||₂ ≤ 1) that gets drowned by a 130-d
log-MFCC vector with much higher magnitude.

Fix: **per-component L2-normalization + scalar weights** before final L2-norm:

    vec = concat( wA·(A/||A||), wB·(B/||B||), wC·(C/||C||), wD·(D/||D||), wE·(E/||E||) )
    fp  = vec / ||vec||

With weights = (1,1,1,1,1), every component contributes EQUALLY to the
final unit vector regardless of native magnitude. With wA = 5 (tone-boost),
tone histogram dominates → minimal-pair protection emerges.

This is the paper §3.5 architecture-fix proposal: keep the linguistically
motivated component design, but normalize each component independently and
weight by typological priority.

Self-test: sweep wA ∈ {1, 2, 5, 10, 20} on the same minimal-pair corpus
and report whether intra-tone < inter-syl at each weight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
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
from utterance_cache import (                                  # noqa: E402
    UtteranceCache,
    cheap_mfcc,
    tone_histogram_from_f0,
    _f0_contour,
)


def _components(audio: np.ndarray, sr: int) -> dict:
    """Re-implemented inline to avoid cross-import (same code as utterance_cache.fingerprint)."""
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
    return {"A_tone": tone_hist, "B_mfcc": mfcc_flat, "C_delta": mfcc_delta_flat,
            "D_f0": f0_stats, "E_energy": energy_stats}


def weighted_fingerprint(audio: np.ndarray, sr: int,
                         weights: dict[str, float] = None) -> np.ndarray:
    """Per-component L2-norm + weight + concat + final L2-norm.

    weights: optional {A_tone, B_mfcc, C_delta, D_f0, E_energy} → float multipliers.
             Default = uniform 1.0 each (so each component contributes equally
             regardless of native dim count).
    """
    weights = weights or {"A_tone": 1.0, "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}
    comps = _components(audio, sr)
    parts = []
    for name in ("A_tone", "B_mfcc", "C_delta", "D_f0", "E_energy"):
        c = comps[name].astype(np.float32)
        n = np.linalg.norm(c)
        if n > 1e-6:
            c = (c / n) * float(weights.get(name, 1.0))
        parts.append(c)
    vec = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


def weighted_fingerprint_from_components(comps: dict, weights: dict[str, float]) -> np.ndarray:
    """Same logic but takes pre-computed components — used by the sweep below."""
    parts = []
    for name in ("A_tone", "B_mfcc", "C_delta", "D_f0", "E_energy"):
        c = comps[name].astype(np.float32)
        n = np.linalg.norm(c)
        if n > 1e-6:
            c = (c / n) * float(weights.get(name, 1.0))
        parts.append(c)
    vec = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


# Minimal-pair sweep: same audio as minimal_pair_eval.py, multiple weight schemes
MINIMAL_PAIR_SYLLABLES = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
    "ta":  ["ta",  "tá",  "tả",  "tạ",  "tã",  "tà"],
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
    ap.add_argument("--out", default="Results/weighted_fingerprint_sweep.json")
    args = ap.parse_args()

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[load] ready, sr={sr}")

    # Synthesize all clips
    print("[synth] generating minimal-pair clips ...")
    audio_by_key = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] synth failed for {v!r}: {e}")
    print(f"[synth] {len(audio_by_key)} clips ready")

    # Pre-compute components once
    comps_by_key = {k: _components(a, sr) for k, a in audio_by_key.items()}

    # Sweep tone-boost weights, keeping all others at 1.0
    schemes = [
        {"label": "uniform_1",         "weights": {"A_tone": 1.0,  "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}},
        {"label": "tone_boost_2",      "weights": {"A_tone": 2.0,  "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}},
        {"label": "tone_boost_5",      "weights": {"A_tone": 5.0,  "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}},
        {"label": "tone_boost_10",     "weights": {"A_tone": 10.0, "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}},
        {"label": "tone_boost_20",     "weights": {"A_tone": 20.0, "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 1.0, "E_energy": 1.0}},
        {"label": "tone+f0_boost",     "weights": {"A_tone": 5.0,  "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 3.0, "E_energy": 1.0}},
        {"label": "tone_only_via_weight", "weights": {"A_tone": 100.0, "B_mfcc": 0.0, "C_delta": 0.0, "D_f0": 0.0, "E_energy": 0.0}},
        {"label": "mfcc_only_via_weight","weights": {"A_tone": 0.0, "B_mfcc": 1.0, "C_delta": 1.0, "D_f0": 0.0, "E_energy": 0.0}},
    ]

    print(f"\n{'scheme':<26}{'intra_tone_cos':>17}{'inter_syl_cos':>16}{'gap (i-i)':>14}{'protect?':>11}")
    print("-" * 86)

    results = []
    for scheme in schemes:
        intra_cosines = []
        for base, variants in MINIMAL_PAIR_SYLLABLES.items():
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    k1, k2 = (base, variants[i]), (base, variants[j])
                    if k1 in comps_by_key and k2 in comps_by_key:
                        v1 = weighted_fingerprint_from_components(comps_by_key[k1], scheme["weights"])
                        v2 = weighted_fingerprint_from_components(comps_by_key[k2], scheme["weights"])
                        intra_cosines.append(float(np.dot(v1, v2)))

        inter_cosines = []
        bases = list(MINIMAL_PAIR_SYLLABLES.keys())
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                for v1n in MINIMAL_PAIR_SYLLABLES[bases[i]]:
                    for v2n in MINIMAL_PAIR_SYLLABLES[bases[j]]:
                        k1, k2 = (bases[i], v1n), (bases[j], v2n)
                        if k1 in comps_by_key and k2 in comps_by_key:
                            v1 = weighted_fingerprint_from_components(comps_by_key[k1], scheme["weights"])
                            v2 = weighted_fingerprint_from_components(comps_by_key[k2], scheme["weights"])
                            inter_cosines.append(float(np.dot(v1, v2)))

        intra_mean = float(np.mean(intra_cosines)) if intra_cosines else 0.0
        inter_mean = float(np.mean(inter_cosines)) if inter_cosines else 0.0
        gap_iim = inter_mean - intra_mean
        protect = gap_iim > 0
        verdict = "✓" if protect else "✗"
        print(f"  {scheme['label']:<24}{intra_mean:>17.4f}{inter_mean:>16.4f}{gap_iim:>+14.4f}{verdict:>11}")
        results.append({
            "label": scheme["label"], "weights": scheme["weights"],
            "intra_tone_cos_mean": round(intra_mean, 4),
            "inter_syl_cos_mean":  round(inter_mean, 4),
            "protect_gap":         round(gap_iim, 4),
            "protects_minimal_pairs": bool(protect),
        })

    print("-" * 86)
    print("\nLegend:")
    print("  ✓ protect: tonal variants more different than syllabic ones (good for minimal pairs)")
    print("  gap (i-i): inter_syl - intra_tone  (positive = protected)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "syllable_groups": MINIMAL_PAIR_SYLLABLES,
        "carrier":         CARRIER,
        "schemes":         results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
