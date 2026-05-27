"""
Fingerprint-component ablation — strip each of the 5 dimension groups,
measure intra-phrase vs inter-phrase cosine on a fixed VIVOS multi-speaker set.

This is the §4 paper table reviewers will look for. Each ablation row =
one fingerprint variant. Each cell = mean cosine within / across phrase
groups.

Five components in the proposed 272-d fingerprint:
  A. Tone histogram        (6 dims)
  B. MFCC temporal-subsampled (130 dims)
  C. MFCC delta (velocity) (130 dims)
  D. F0 statistics         (4 dims)
  E. Energy statistics     (2 dims)

Ablations:
  - full       : A + B + C + D + E   (proposed)
  - no tone    : B + C + D + E
  - no MFCC    : A + D + E             (only tonal + global stats)
  - no delta   : A + B + D + E
  - no F0_stats: A + B + C + E
  - no energy  : A + B + C + D
  - mfcc_only_subsampled : B           (segment trajectory alone)
  - tone_only  : A                     (suprasegmental alone)

For each variant we compute:
  intra_cos = mean cos(fp_i, fp_j) where i, j are same-phrase, different-speaker
  inter_cos = mean cos(fp_i, fp_j) where i, j are different-phrase
  gap       = intra_cos - inter_cos  (larger is better — more discriminative)

A "good" fingerprint has intra > inter by a clean margin (>= 0.05 typical).
A degenerate fingerprint has gap ≈ 0 or negative.

VIVOS provides ~46 train speakers × ~30 utterances each. We use the same
utterance text across multiple speakers (matched by VIVOSSPK*_R001 etc) to
form intra-phrase pairs, and across different utterances for inter-phrase.

Output: Results/fingerprint_ablation.json + console table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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


PROJECT = HERE.parent
VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"


# ----------- Component extractors -----------

def _components(audio: np.ndarray, sr: int) -> dict:
    """Return all 5 fingerprint components for an audio sample."""
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
    n_chunks = 10
    mfcc_subsampled = UtteranceCache._temporal_subsample(mfcc, n_chunks=n_chunks)
    mfcc_flat       = mfcc_subsampled.flatten(order="F")
    mfcc_delta      = np.diff(mfcc_subsampled, axis=1, prepend=mfcc_subsampled[:, :1])
    mfcc_delta_flat = mfcc_delta.flatten(order="F")

    rms_all = float(np.sqrt((audio ** 2).mean() + 1e-12))
    e_bounds = np.linspace(0, len(audio), 9, dtype=int)
    chunk_rms = [float(np.sqrt((audio[e_bounds[i]:e_bounds[i+1]] ** 2).mean() + 1e-12))
                 for i in range(8)]
    energy_stats = np.array([rms_all, max(chunk_rms)], dtype=np.float32)

    return {
        "A_tone":    tone_hist,
        "B_mfcc":    mfcc_flat,
        "C_delta":   mfcc_delta_flat,
        "D_f0":      f0_stats,
        "E_energy":  energy_stats,
    }


def _build_fingerprint(components: dict, keep: tuple[str, ...]) -> np.ndarray:
    """L2-normalize the concatenation of the kept components."""
    parts = [components[k] for k in keep if k in components and len(components[k]) > 0]
    if not parts:
        return np.zeros(1, dtype=np.float32)
    vec = np.concatenate(parts).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


# ----------- Pair sampling -----------

def _load_vivos_pairs(max_per_phrase: int = 4) -> tuple[list, list]:
    """Return (intra_pairs, inter_pairs).
       intra_pairs: list of (path1, path2) for same-phrase different-speaker.
       inter_pairs: list of (path1, path2) for different-phrase.

    VIVOS naming convention: VIVOSSPK<NN>_R<NNN>.wav — phrase index after _R.
    So phrase ids match across speakers (R001 is the same phrase).
    """
    prompts_file = VIVOS_TRAIN / "prompts.txt"
    text_by_id: dict[str, str] = {}
    if prompts_file.exists():
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            utt_id, _, text = line.partition(" ")
            text_by_id[utt_id.strip()] = text.strip().lower()

    waves_root = VIVOS_TRAIN / "waves"
    if not waves_root.exists():
        return [], []
    # Group recordings by their TEXT (not by phrase index — different speakers may
    # have R001 saying *different* texts in VIVOS).
    paths_by_text: dict[str, list[Path]] = defaultdict(list)
    for spk_dir in sorted(waves_root.iterdir()):
        if not spk_dir.is_dir(): continue
        for wav in sorted(spk_dir.glob("*.wav")):
            utt_id = wav.stem
            text = text_by_id.get(utt_id)
            if text is None: continue
            paths_by_text[text].append(wav)

    intra_pairs = []
    for text, paths in paths_by_text.items():
        if len(paths) < 2:
            continue
        # take up to max_per_phrase pairs per text
        for i in range(min(len(paths), max_per_phrase)):
            for j in range(i + 1, min(len(paths), max_per_phrase)):
                intra_pairs.append((paths[i], paths[j]))

    # Inter-phrase: sample pairs from different texts.
    rng = np.random.default_rng(42)
    keys = list(paths_by_text.keys())
    inter_pairs = []
    n_inter = min(len(intra_pairs), 200)
    while len(inter_pairs) < n_inter:
        a, b = rng.choice(len(keys), 2, replace=False)
        if not paths_by_text[keys[a]] or not paths_by_text[keys[b]]:
            continue
        p1 = paths_by_text[keys[a]][rng.integers(len(paths_by_text[keys[a]]))]
        p2 = paths_by_text[keys[b]][rng.integers(len(paths_by_text[keys[b]]))]
        inter_pairs.append((p1, p2))
    return intra_pairs, inter_pairs


# ----------- Ablation runner -----------

ABLATIONS = [
    ("full",                  ("A_tone", "B_mfcc", "C_delta", "D_f0", "E_energy")),
    ("no_tone",               ("B_mfcc", "C_delta", "D_f0", "E_energy")),
    ("no_mfcc",               ("A_tone", "D_f0", "E_energy")),
    ("no_delta",              ("A_tone", "B_mfcc", "D_f0", "E_energy")),
    ("no_f0_stats",           ("A_tone", "B_mfcc", "C_delta", "E_energy")),
    ("no_energy",             ("A_tone", "B_mfcc", "C_delta", "D_f0")),
    ("mfcc_subsampled_only",  ("B_mfcc",)),
    ("tone_only",             ("A_tone",)),
    ("mfcc_means_only",       ("B_mfcc", "D_f0")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs",   type=int, default=100, help="Cap intra+inter pair count for speed")
    ap.add_argument("--out",         default="Results/fingerprint_ablation.json")
    args = ap.parse_args()

    print("[data] loading VIVOS pair lists ...")
    intra_pairs, inter_pairs = _load_vivos_pairs(max_per_phrase=4)
    n_intra = min(len(intra_pairs), args.max_pairs)
    n_inter = min(len(inter_pairs), args.max_pairs)
    intra_pairs = intra_pairs[:n_intra]
    inter_pairs = inter_pairs[:n_inter]
    print(f"[data] intra-phrase pairs: {n_intra}   inter-phrase pairs: {n_inter}")

    # Cache components per file to avoid re-computing
    unique_paths = set()
    for a, b in intra_pairs + inter_pairs:
        unique_paths.add(a); unique_paths.add(b)
    print(f"[data] extracting components for {len(unique_paths)} unique files...")
    components_per_path: dict[Path, dict] = {}
    t0 = time.perf_counter()
    for i, p in enumerate(sorted(unique_paths), start=1):
        audio, sr = sf.read(str(p), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        components_per_path[p] = _components(audio, sr)
        if i % 20 == 0:
            print(f"  [{i}/{len(unique_paths)}]  {(time.perf_counter()-t0):.0f}s")
    print(f"[data] components ready in {(time.perf_counter()-t0):.0f}s")

    print(f"\n{'ablation':<26}{'dims':>6}{'intra_cos':>12}{'inter_cos':>12}{'gap':>10}{'separable?':>13}")
    print("-" * 79)

    results = []
    for name, keep in ABLATIONS:
        intra_cosines = []
        for p1, p2 in intra_pairs:
            v1 = _build_fingerprint(components_per_path[p1], keep)
            v2 = _build_fingerprint(components_per_path[p2], keep)
            if len(v1) == len(v2) and len(v1) > 0:
                intra_cosines.append(float(np.dot(v1, v2)))
        inter_cosines = []
        for p1, p2 in inter_pairs:
            v1 = _build_fingerprint(components_per_path[p1], keep)
            v2 = _build_fingerprint(components_per_path[p2], keep)
            if len(v1) == len(v2) and len(v1) > 0:
                inter_cosines.append(float(np.dot(v1, v2)))

        intra_mean = float(np.mean(intra_cosines)) if intra_cosines else 0.0
        inter_mean = float(np.mean(inter_cosines)) if inter_cosines else 0.0
        intra_std  = float(np.std(intra_cosines))  if intra_cosines else 0.0
        inter_std  = float(np.std(inter_cosines))  if inter_cosines else 0.0
        gap        = intra_mean - inter_mean
        # Separability: 3-sigma rule. Clean gap if intra_mean - inter_mean > 3*max(std).
        separable  = gap > 3.0 * max(intra_std, inter_std)
        # Dim count
        dims = sum(len(components_per_path[next(iter(unique_paths))].get(k, [])) for k in keep)
        results.append({
            "ablation":    name,
            "components":  list(keep),
            "dims":        int(dims),
            "intra_cos_mean": round(intra_mean, 4),
            "intra_cos_std":  round(intra_std, 4),
            "inter_cos_mean": round(inter_mean, 4),
            "inter_cos_std":  round(inter_std, 4),
            "gap":            round(gap, 4),
            "separable_3sigma": bool(separable),
        })
        marker = "✓" if separable else "✗"
        print(f"  {name:<24}{dims:>6}{intra_mean:>12.4f}{inter_mean:>12.4f}{gap:>10.4f}{marker:>13}")

    print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_intra_pairs": n_intra,
        "n_inter_pairs": n_inter,
        "ablations":     results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
