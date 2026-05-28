"""
Dynamic Time Warping (DTW) classical baseline for speech-retrieval caching.

Response to ACL reviewer Q5/W: "Limited discussion of speech retrieval using SSL
representations, spoken term detection, and robust hashing alternatives beyond
Shazam-style methods." DTW over MFCC sequences is the textbook pre-neural
baseline for variable-length acoustic similarity (Sakoe & Chiba 1978;
Rabiner & Juang 1993, Ch.4). We measure it on the same two pools used by the
two-stream voting evaluation:

  (A) 200 VIVOS positive pairs (same prompt text, different speakers)
      + 200 VIVOS negative pairs (different prompt texts)
        — measures cross-speaker phrase-recognition F1.
  (B) 30 tonal minimal-pair clips synthesized via MMS-TTS-vie
      (5 syllables x 6 Vietnamese tones) under the existence-rule cache
      protocol of two_stream_existence_eval.py
        — measures whether DTW tells *ma* from *má*.

DTW pipeline:
  1. Resample to 16 kHz, extract 13-d MFCC sequence (T_i frames) via cheap_mfcc.
  2. Per-frame MFCC normalization (cosine distance, equivalent to L2 on L2-normed).
  3. Classical DP recurrence:
       D[i,j] = d(x_i, y_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
     with d = 1 - cosine, banded to |i*T2/T1 - j| <= radius for tractability.
  4. Similarity sim = exp(-D[T1,T2] / (T1 + T2)).
  5. Pick tau* on VIVOS pairs maximizing F1; report VIVOS metrics and tonal
     minimal-pair correct-miss rate at tau*.

Pure NumPy. Soft-depends only on soundfile (already a project dep) and
transformers+MMS-TTS-vie for the minimal-pair audio (synthesis is identical
to two_stream_existence_eval.py so a side-by-side comparison is apples-to-apples).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

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

import numpy as np                                          # noqa: E402
import soundfile as sf                                      # noqa: E402

from utterance_cache import cheap_mfcc                       # noqa: E402
from two_stream_voting import MINIMAL_PAIR_SYLLABLES, CARRIER, _synth  # noqa: E402


VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

SR_TARGET    = 16000
N_MFCC       = 13
N_PAIRS      = 200
DTW_RADIUS_F = 0.20      # Sakoe-Chiba band radius as fraction of max(T1, T2)


# -------------------- Audio loading / MFCC --------------------

def _load_audio(p: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(p), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), int(sr)


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_TARGET:
        return audio
    n_new = int(round(len(audio) * SR_TARGET / sr))
    return np.interp(np.linspace(0, len(audio), n_new, endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


def _mfcc_seq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return MFCC matrix shape (T, n_mfcc), per-frame L2-normalized."""
    audio = _resample_to_16k(audio, sr)
    mfcc = cheap_mfcc(audio, SR_TARGET, n_mfcc=N_MFCC)   # (n_mfcc, T)
    mfcc = mfcc.T.astype(np.float32)                     # (T, n_mfcc)
    norms = np.linalg.norm(mfcc, axis=1, keepdims=True) + 1e-9
    return mfcc / norms


# -------------------- DTW core (pure NumPy) --------------------

def dtw_distance(x: np.ndarray, y: np.ndarray, radius_frac: float = DTW_RADIUS_F) -> float:
    """Classical DTW with Sakoe-Chiba band; returns accumulated cost D[T1,T2].

    x: (T1, d) L2-normed frames; y: (T2, d) L2-normed frames.
    d(i,j) = 1 - cos = 1 - x_i . y_j.
    Band radius = max(1, int(radius_frac * max(T1, T2))).
    Cells outside the band are inf so the path is forced near the diagonal,
    which is both the standard speech-DTW choice and necessary for tractability
    on multi-second audio (T ~ 200-800 frames at 10 ms hop).
    """
    T1, T2 = x.shape[0], y.shape[0]
    if T1 == 0 or T2 == 0:
        return float("inf")
    radius = max(1, int(radius_frac * max(T1, T2)))
    # Cosine-distance matrix only within band; outside set to +inf
    INF = 1e18
    D = np.full((T1 + 1, T2 + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    # Pairwise distance row-by-row to bound memory at O(T)
    for i in range(1, T1 + 1):
        # band column range for this row
        j_center = (i * T2) // T1
        j_lo = max(1, j_center - radius)
        j_hi = min(T2, j_center + radius)
        xi = x[i - 1]
        # vectorize within band
        sub_y = y[j_lo - 1:j_hi]
        d_row = 1.0 - sub_y @ xi   # cosine distance, in [0, 2]
        for k, j in enumerate(range(j_lo, j_hi + 1)):
            cost = d_row[k]
            prev = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            if prev < INF:
                D[i, j] = cost + prev
    return float(D[T1, T2])


def dtw_similarity(x: np.ndarray, y: np.ndarray, radius_frac: float = DTW_RADIUS_F) -> float:
    d = dtw_distance(x, y, radius_frac=radius_frac)
    T = x.shape[0] + y.shape[0]
    if T == 0 or not np.isfinite(d):
        return 0.0
    return float(np.exp(-d / T))


# -------------------- VIVOS pair sampling (mirrors threshold_sweep.py) --------------------

def _load_vivos_text_groups() -> dict[str, list[tuple[Path, str]]]:
    prompts_file = VIVOS_TRAIN / "prompts.txt"
    text_by_id: dict[str, str] = {}
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        text_by_id[utt_id.strip()] = text.strip().lower()
    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for spk_dir in sorted((VIVOS_TRAIN / "waves").iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            text = text_by_id.get(wav.stem)
            if text is None:
                continue
            groups[text].append((wav, spk))
    return groups


def _build_pairs(n_per_class: int = N_PAIRS, seed: int = 0
                 ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], dict]:
    groups = _load_vivos_text_groups()
    rng = np.random.default_rng(seed)

    pos_pairs: list[tuple[Path, Path]] = []
    multi_spk_groups = 0
    for text, entries in groups.items():
        spk_to_paths: dict[str, list[Path]] = defaultdict(list)
        for p, s in entries:
            spk_to_paths[s].append(p)
        if len(spk_to_paths) < 2:
            continue
        multi_spk_groups += 1
        spks = list(spk_to_paths.keys())
        rng.shuffle(spks)
        for i in range(len(spks) - 1):
            for j in range(i + 1, len(spks)):
                pos_pairs.append((spk_to_paths[spks[i]][0], spk_to_paths[spks[j]][0]))
                if len(pos_pairs) >= n_per_class:
                    break
            if len(pos_pairs) >= n_per_class:
                break
        if len(pos_pairs) >= n_per_class:
            break

    keys = list(groups.keys())
    neg_pairs: list[tuple[Path, Path]] = []
    guard = 0
    while len(neg_pairs) < n_per_class and guard < n_per_class * 20:
        guard += 1
        ai, bi = rng.choice(len(keys), 2, replace=False)
        ga, gb = groups[keys[ai]], groups[keys[bi]]
        if not ga or not gb:
            continue
        p1 = ga[rng.integers(len(ga))][0]
        p2 = gb[rng.integers(len(gb))][0]
        neg_pairs.append((p1, p2))

    return pos_pairs, neg_pairs, {
        "n_unique_texts": len(groups),
        "n_multi_speaker_groups": multi_spk_groups,
        "n_pos_pairs": len(pos_pairs),
        "n_neg_pairs": len(neg_pairs),
    }


# -------------------- Score-and-sweep --------------------

def _score_pairs(pairs: list[tuple[Path, Path]], mfcc_cache: dict[Path, np.ndarray]
                 ) -> tuple[list[float], list[float]]:
    """Return (similarities, latencies_ms_per_pair)."""
    sims, lats = [], []
    for i, (p1, p2) in enumerate(pairs):
        x = mfcc_cache.get(p1)
        if x is None:
            a, sr = _load_audio(p1)
            x = _mfcc_seq(a, sr)
            mfcc_cache[p1] = x
        y = mfcc_cache.get(p2)
        if y is None:
            a, sr = _load_audio(p2)
            y = _mfcc_seq(a, sr)
            mfcc_cache[p2] = y
        t0 = time.perf_counter()
        s = dtw_similarity(x, y)
        t1 = time.perf_counter()
        sims.append(s)
        lats.append((t1 - t0) * 1000.0)
        if (i + 1) % 50 == 0:
            print(f"    [pair {i+1}/{len(pairs)}] s={s:.4f}  dtw_ms={lats[-1]:.1f}", flush=True)
    return sims, lats


def _best_f1_threshold(pos_sims: list[float], neg_sims: list[float]
                       ) -> tuple[float, float, float, float, int, int]:
    """Sweep thresholds; return (tau*, P, R, F1, TP, FP)."""
    cand = sorted(set(pos_sims + neg_sims))
    best = (-1.0, 0.0, 0.0, 0.0, 0, 0)
    for tau in cand:
        tp = sum(1 for s in pos_sims if s >= tau)
        fp = sum(1 for s in neg_sims if s >= tau)
        fn = len(pos_sims) - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[3]:
            best = (tau, p, r, f1, tp, fp)
    return best


# -------------------- Minimal-pair (tonal) eval --------------------

def _tonal_minimal_pair_eval(tau: float, radius_frac: float = DTW_RADIUS_F) -> dict:
    """Mirror two_stream_existence_eval._eval_rule exp2 with DTW-similarity gate.

    Store ngang-tone clip for each base syllable; query the OTHER 5 tones;
    hit iff sim(query, stored) >= tau. correct-miss = miss on tonal variant.
    Self-lookup test confirms recall=100% at the same threshold (sim=1.0 for
    identical sequences).
    """
    print("[load] MMS-TTS-vie (for minimal-pair synthesis) ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    audio_by_key = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] {v}: {e}")
    print(f"[synth] {len(audio_by_key)} clips")

    # Pre-extract MFCC seqs (~30 clips * a few hundred frames each)
    mfcc_by_key = {k: _mfcc_seq(a, sr) for k, a in audio_by_key.items()}

    # Stored: ngang-tone variant of each base; queried: every OTHER tone of same base
    stored: dict[str, np.ndarray] = {}
    for base in MINIMAL_PAIR_SYLLABLES:
        ngang_key = (base, MINIMAL_PAIR_SYLLABLES[base][0])
        if ngang_key in mfcc_by_key:
            stored[base] = mfcc_by_key[ngang_key]

    # self-lookup recall (sanity)
    self_hits = 0
    for k, x in mfcc_by_key.items():
        # store-and-query each separately
        s = dtw_similarity(x, x)
        if s >= tau:
            self_hits += 1

    false_hits = 0
    correct_misses = 0
    total = 0
    per_query = []
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        if base not in stored:
            continue
        st = stored[base]
        for v in variants[1:]:                  # skip ngang (already stored)
            k = (base, v)
            if k not in mfcc_by_key:
                continue
            x = mfcc_by_key[k]
            s = dtw_similarity(x, st)
            total += 1
            hit = s >= tau
            if hit:
                false_hits += 1
            else:
                correct_misses += 1
            per_query.append({"base": base, "tone": v, "sim_vs_ngang": round(s, 4), "hit": bool(hit)})

    return {
        "tau_used":                              round(tau, 6),
        "n_clips_synthesized":                   len(audio_by_key),
        "self_lookups":                          len(mfcc_by_key),
        "self_hits":                             self_hits,
        "self_hit_rate":                         round(self_hits / max(len(mfcc_by_key), 1), 4),
        "tonal_variant_lookups":                 total,
        "tonal_variant_correct_misses":          correct_misses,
        "tonal_variant_correct_miss_rate":       round(correct_misses / max(total, 1), 4),
        "tonal_variant_false_hits":              false_hits,
        "tonal_variant_false_hit_rate":          round(false_hits / max(total, 1), 4),
        "per_query":                             per_query,
    }


# -------------------- Driver --------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs",     type=int, default=N_PAIRS)
    ap.add_argument("--radius_frac", type=float, default=DTW_RADIUS_F)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--skip_minpair", action="store_true",
                    help="Skip the MMS-TTS-vie minimal-pair eval (e.g. when network unavailable)")
    ap.add_argument("--out",         default=str(PROJECT / "Results" / "dtw_baseline.json"))
    args = ap.parse_args()

    print(f"[config] n_pairs/class={args.n_pairs} radius_frac={args.radius_frac} seed={args.seed}")
    print("[pairs] building VIVOS positive/negative pair pool ...")
    pos_pairs, neg_pairs, meta = _build_pairs(args.n_pairs, seed=args.seed)
    print(f"[pairs] {meta}")

    mfcc_cache: dict[Path, np.ndarray] = {}
    print("\n[score] DTW on POSITIVE pairs (same text, different speakers) ...")
    t0 = time.perf_counter()
    pos_sims, pos_lats = _score_pairs(pos_pairs, mfcc_cache)
    pos_time = time.perf_counter() - t0
    print(f"  done in {pos_time:.1f}s  "
          f"sim mean={np.mean(pos_sims):.4f}  median={np.median(pos_sims):.4f}")

    print("\n[score] DTW on NEGATIVE pairs (different texts) ...")
    t0 = time.perf_counter()
    neg_sims, neg_lats = _score_pairs(neg_pairs, mfcc_cache)
    neg_time = time.perf_counter() - t0
    print(f"  done in {neg_time:.1f}s  "
          f"sim mean={np.mean(neg_sims):.4f}  median={np.median(neg_sims):.4f}")

    all_lats = pos_lats + neg_lats

    print("\n[sweep] best-F1 threshold over union of observed sims ...")
    tau, prec, rec, f1, tp, fp = _best_f1_threshold(pos_sims, neg_sims)
    print(f"  tau* = {tau:.4f}")
    print(f"  P    = {prec:.4f}   R = {rec:.4f}   F1 = {f1:.4f}")
    print(f"  TP   = {tp} / {len(pos_sims)}    FP = {fp} / {len(neg_sims)}")

    vivos_block = {
        "n_pos_pairs": len(pos_sims),
        "n_neg_pairs": len(neg_sims),
        "pos_sim_mean":   round(float(np.mean(pos_sims)),   6),
        "pos_sim_median": round(float(np.median(pos_sims)), 6),
        "pos_sim_std":    round(float(np.std(pos_sims)),    6),
        "neg_sim_mean":   round(float(np.mean(neg_sims)),   6),
        "neg_sim_median": round(float(np.median(neg_sims)), 6),
        "neg_sim_std":    round(float(np.std(neg_sims)),    6),
        "best_f1_threshold": round(tau, 6),
        "precision":      round(prec, 6),
        "recall":         round(rec,  6),
        "f1":             round(f1,   6),
        "true_positives":  tp,
        "false_positives": fp,
        "dtw_ms_per_pair_mean":   round(float(np.mean(all_lats)),   3),
        "dtw_ms_per_pair_median": round(float(np.median(all_lats)), 3),
        "dtw_ms_per_pair_p95":    round(float(np.percentile(all_lats, 95)), 3),
        "dtw_ms_per_pair_p99":    round(float(np.percentile(all_lats, 99)), 3),
        "dtw_total_seconds": round(pos_time + neg_time, 3),
    }

    if args.skip_minpair:
        minpair_block: Optional[dict] = None
        print("\n[minpair] skipped (--skip_minpair)")
    else:
        print(f"\n[minpair] DTW on synthesized tonal minimal pairs at tau* = {tau:.4f} ...")
        minpair_block = _tonal_minimal_pair_eval(tau, radius_frac=args.radius_frac)
        print(f"  self-lookup:  {minpair_block['self_hits']}/{minpair_block['self_lookups']}")
        print(f"  correct-miss: {minpair_block['tonal_variant_correct_misses']}/"
              f"{minpair_block['tonal_variant_lookups']}  "
              f"({minpair_block['tonal_variant_correct_miss_rate']*100:.1f}%)")
        print(f"  false-hits:   {minpair_block['tonal_variant_false_hits']}/"
              f"{minpair_block['tonal_variant_lookups']}  "
              f"({minpair_block['tonal_variant_false_hit_rate']*100:.1f}%)")

    out = {
        "config": {
            "n_mfcc":               N_MFCC,
            "sr_target":            SR_TARGET,
            "dtw_radius_frac":      args.radius_frac,
            "n_pairs_per_class":    args.n_pairs,
            "seed":                 args.seed,
        },
        "vivos_pair_meta":          meta,
        "vivos_pair_eval":          vivos_block,
        "tonal_minimal_pair_eval":  minpair_block,
        "note": (
            "DTW with cosine distance over 13-d MFCC + Sakoe-Chiba band. "
            "sim = exp(-D / (T1+T2)). Operating-point tau* picked on VIVOS pairs by max-F1. "
            "Minimal-pair correct-miss measured at the SAME tau* — this is the head-to-head "
            "comparison vs two_stream_existence_eval (existence-rule and argmax-equality rules)."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
