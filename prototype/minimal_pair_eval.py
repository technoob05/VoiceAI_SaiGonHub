"""
Tonal minimal-pair eval — validate the tone-histogram component on Vi homonyms.

The fingerprint ablation on VIVOS read speech (50 intra + 50 inter pairs)
showed that tone histogram contributes ~0 measurable gap. Re-positioning
claim: tone histogram is a TYPOLOGY SAFETY NET for tonal minimal pairs
(words differing only in tone, e.g., ma/má/mả/mạ/mã/mà).

This script tests that claim by:
  1. Synthesizing 6 tonal variants of each of N minimal-pair syllables
     via Vi MMS-TTS (facebook/mms-tts-vie).
  2. Computing the full 272-d fingerprint and the no-tone (266-d) variant.
  3. Measuring intra-syllable (same syllable, different tone) cosine vs
     inter-syllable (different syllable) cosine for each variant.

PASS criterion: full fingerprint has lower intra-syllable cosine than
no-tone variant (meaning tone histogram DECREASES same-syllable similarity
when the tone differs, i.e., adds disambiguation).

We use carrier phrases "tôi nói X" to give the TTS enough context to
produce natural tonal contours.

Output: Results/minimal_pair_eval.json + console table.
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


# Carrier-phrase template — gives TTS context to render tone naturally.
# We use "nói X" ("say X") which is a common Vi sentence frame.
CARRIER = "nói {syl}"

# Minimal-pair syllables — each row has 6 tonal variants of the same base.
# Picked to be all *real* Vietnamese words (or near-words) to avoid silly TTS.
MINIMAL_PAIR_SYLLABLES = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
    "ta":  ["ta",  "tá",  "tả",  "tạ",  "tã",  "tà"],
}


def _components(audio: np.ndarray, sr: int) -> dict:
    """Same as fingerprint_ablation._components — re-implemented here to avoid cross-import."""
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
    mfcc_subsampled = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
    mfcc_flat       = mfcc_subsampled.flatten(order="F")
    mfcc_delta      = np.diff(mfcc_subsampled, axis=1, prepend=mfcc_subsampled[:, :1])
    mfcc_delta_flat = mfcc_delta.flatten(order="F")
    rms_all = float(np.sqrt((audio ** 2).mean() + 1e-12))
    e_bounds = np.linspace(0, len(audio), 9, dtype=int)
    chunk_rms = [float(np.sqrt((audio[e_bounds[i]:e_bounds[i+1]] ** 2).mean() + 1e-12))
                 for i in range(8)]
    energy_stats = np.array([rms_all, max(chunk_rms)], dtype=np.float32)
    return {"A_tone": tone_hist, "B_mfcc": mfcc_flat, "C_delta": mfcc_delta_flat,
            "D_f0": f0_stats, "E_energy": energy_stats}


def _build_fp(components: dict, keep: tuple) -> np.ndarray:
    parts = [components[k] for k in keep if k in components and len(components[k]) > 0]
    if not parts:
        return np.zeros(1, dtype=np.float32)
    vec = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


def synth_audio(text: str, tok, model, sr: int) -> np.ndarray:
    import torch
    inputs = tok(text, return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform
    return wav.squeeze().cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Results/minimal_pair_eval.json")
    args = ap.parse_args()

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[load] ready, sr={sr}")

    # ---- Synthesize all minimal-pair audios ----
    print("\n[synth] minimal-pair audios via 'nói X' carrier frame")
    audio_by_key: dict[tuple, np.ndarray] = {}    # (base_syl, full_tonal_form) -> audio
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            text = CARRIER.format(syl=v)
            try:
                a = synth_audio(text, tok, model, sr)
                audio_by_key[(base, v)] = a
            except Exception as e:
                print(f"  [warn] synth failed for {text!r}: {e}")
    print(f"[synth] generated {len(audio_by_key)} audio clips")

    # ---- Extract components per clip ----
    comps_by_key = {k: _components(a, sr) for k, a in audio_by_key.items()}

    # ---- Pairwise cosine: intra-syllable (same base, different tone) vs inter-syllable ----
    variants_to_test = {
        "full":         ("A_tone", "B_mfcc", "C_delta", "D_f0", "E_energy"),
        "no_tone":      ("B_mfcc", "C_delta", "D_f0", "E_energy"),
        "no_mfcc":      ("A_tone", "D_f0", "E_energy"),
        "tone_only":    ("A_tone",),
        "mfcc_only":    ("B_mfcc",),
    }
    print(f"\n[eval] computing pairwise cosines on {len(audio_by_key)} clips × {len(variants_to_test)} variants")
    print(f"{'variant':<14}{'dims':>6}{'intra_tone_cos':>17}{'intra_tone_std':>17}{'inter_syl_cos':>16}{'inter_syl_std':>16}{'gap':>10}{'protect?':>10}")
    print("-" * 110)

    results = []
    for name, keep in variants_to_test.items():
        # Intra-syllable: same base, different tone (typology safety-net test target).
        intra_cosines = []
        for base, variants in MINIMAL_PAIR_SYLLABLES.items():
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    k1 = (base, variants[i])
                    k2 = (base, variants[j])
                    if k1 in comps_by_key and k2 in comps_by_key:
                        v1 = _build_fp(comps_by_key[k1], keep)
                        v2 = _build_fp(comps_by_key[k2], keep)
                        if len(v1) == len(v2) and len(v1) > 0:
                            intra_cosines.append(float(np.dot(v1, v2)))

        # Inter-syllable: different base entirely.
        inter_cosines = []
        bases = list(MINIMAL_PAIR_SYLLABLES.keys())
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                for v1n in MINIMAL_PAIR_SYLLABLES[bases[i]]:
                    for v2n in MINIMAL_PAIR_SYLLABLES[bases[j]]:
                        k1 = (bases[i], v1n)
                        k2 = (bases[j], v2n)
                        if k1 in comps_by_key and k2 in comps_by_key:
                            v1 = _build_fp(comps_by_key[k1], keep)
                            v2 = _build_fp(comps_by_key[k2], keep)
                            if len(v1) == len(v2) and len(v1) > 0:
                                inter_cosines.append(float(np.dot(v1, v2)))

        intra_mean = float(np.mean(intra_cosines)) if intra_cosines else 0.0
        intra_std  = float(np.std(intra_cosines))  if intra_cosines else 0.0
        inter_mean = float(np.mean(inter_cosines)) if inter_cosines else 0.0
        inter_std  = float(np.std(inter_cosines))  if inter_cosines else 0.0
        # Lower intra-tone cosine = better minimal-pair protection (tonal variants discriminated)
        protect_score = inter_mean - intra_mean   # positive: variants are MORE different than syllables
        dims = sum(len(comps_by_key[next(iter(comps_by_key))].get(k, [])) for k in keep)

        results.append({
            "variant":            name,
            "components":         list(keep),
            "dims":               int(dims),
            "intra_tone_cos_mean": round(intra_mean, 4),
            "intra_tone_cos_std":  round(intra_std,  4),
            "inter_syl_cos_mean":  round(inter_mean, 4),
            "inter_syl_cos_std":   round(inter_std,  4),
            "protect_score":      round(protect_score, 4),
        })

        # Heuristic verdict: GOOD if intra-tone < inter-syl (tonal variants discriminated)
        verdict = "✓" if intra_mean < inter_mean else "✗"
        print(f"  {name:<12}{dims:>6}{intra_mean:>17.4f}{intra_std:>17.4f}{inter_mean:>16.4f}{inter_std:>16.4f}{protect_score:>+10.4f}{verdict:>10}")

    print("-" * 110)
    print("\nLegend:")
    print("  intra_tone_cos: cosine between same-syllable-different-tone pairs (ma vs má)")
    print("  inter_syl_cos:  cosine between different-syllable pairs (ma* vs ba*)")
    print("  ✓ protect:      intra_tone < inter_syl → tonal variants more different than syllables (good)")
    print("  ✗ no protect:   intra_tone >= inter_syl → tonal variants confused with same syllable (bad)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_syllables_synthed": len(audio_by_key),
        "syllable_groups":     {k: v for k, v in MINIMAL_PAIR_SYLLABLES.items()},
        "carrier_template":    CARRIER,
        "variants":            results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
