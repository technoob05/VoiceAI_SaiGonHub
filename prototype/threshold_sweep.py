"""
Threshold sweep for the cache-hit decision — measure precision / recall / F1 as
a function of cosine threshold for both fingerprint architectures.

Two architectures swept:
  1. Single-vector UtteranceCache  → sweep τ ∈ [0.70, 0.99] step 0.02
  2. Two-stream TwoStreamCache     → grid τ_tone ∈ {0.70, 0.85, 0.95}
                                        × τ_seg  ∈ {0.85, 0.93, 0.97}

Labels come from VIVOS train:
  POSITIVE: same prompt text, DIFFERENT speakers
            (intra-phrase, cross-speaker — what the cache SHOULD hit)
  NEGATIVE: different prompt texts
            (inter-phrase — what the cache MUST NOT hit)

For each pair we extract a fingerprint with each architecture, score it, then
sweep thresholds. At each operating point we count:
    TP  = positive pair scoring ≥ τ (correct hit)
    FP  = negative pair scoring ≥ τ (wrong-translation risk!)
    FN  = positive pair scoring < τ  (missed reuse)
    TN  = negative pair scoring < τ
Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 standard.

For two-stream the decision is: tone_score ≥ τ_tone AND segment_score ≥ τ_seg.
(We do NOT enforce "best_id matches" here because pairs are 1-vs-1, not
N-vs-cache — the AND-gate is the actual decision being thresholded.)

Output: Results/threshold_sweep.json + console pretty-print of operating points.
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

import numpy as np                                                  # noqa: E402
import soundfile as sf                                              # noqa: E402

from utterance_cache import UtteranceCache                          # noqa: E402
from two_stream_voting import TwoStreamCache                        # noqa: E402


PROJECT     = HERE.parent
VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

CURRENT_SINGLE_TAU       = 0.95
CURRENT_TWOSTREAM_TONE   = 0.85
CURRENT_TWOSTREAM_SEG    = 0.93


# ---------------- Pair sampling ----------------

def _load_vivos_text_groups() -> dict[str, list[tuple[Path, str]]]:
    """Return dict: text -> list of (wav_path, speaker_id)."""
    prompts_file = VIVOS_TRAIN / "prompts.txt"
    text_by_id: dict[str, str] = {}
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        text_by_id[utt_id.strip()] = text.strip().lower()

    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    waves_root = VIVOS_TRAIN / "waves"
    for spk_dir in sorted(waves_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            text = text_by_id.get(wav.stem)
            if text is None:
                continue
            groups[text].append((wav, spk))
    return groups


def _build_pairs(target_per_class: int = 200, seed: int = 0
                 ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], dict]:
    groups = _load_vivos_text_groups()
    rng = np.random.default_rng(seed)

    # POSITIVE: same text, different speakers
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
        # one pair per group between two random distinct speakers (cap saturation)
        rng.shuffle(spks)
        for i in range(len(spks) - 1):
            for j in range(i + 1, len(spks)):
                p1 = spk_to_paths[spks[i]][0]
                p2 = spk_to_paths[spks[j]][0]
                pos_pairs.append((p1, p2))
                if len(pos_pairs) >= target_per_class:
                    break
            if len(pos_pairs) >= target_per_class:
                break
        if len(pos_pairs) >= target_per_class:
            break

    # NEGATIVE: different texts (any speaker)
    keys = list(groups.keys())
    neg_pairs: list[tuple[Path, Path]] = []
    guard = 0
    while len(neg_pairs) < target_per_class and guard < target_per_class * 20:
        guard += 1
        ai, bi = rng.choice(len(keys), 2, replace=False)
        ga, gb = groups[keys[ai]], groups[keys[bi]]
        if not ga or not gb:
            continue
        p1 = ga[rng.integers(len(ga))][0]
        p2 = gb[rng.integers(len(gb))][0]
        neg_pairs.append((p1, p2))

    meta = {
        "n_unique_texts":         len(groups),
        "n_multi_speaker_groups": multi_spk_groups,
        "n_pos_pairs":            len(pos_pairs),
        "n_neg_pairs":            len(neg_pairs),
    }
    return pos_pairs, neg_pairs, meta


# ---------------- Fingerprint helpers ----------------

_FP_SINGLE_CACHE: dict[Path, np.ndarray] = {}
_FP_TWOSTREAM_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

_single_cache_helper   = UtteranceCache(max_entries=1, cosine_hit=1.1)
_twostream_cache_helper = TwoStreamCache()


def _load_audio(p: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(p), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), int(sr)


def _fp_single(p: Path) -> np.ndarray:
    v = _FP_SINGLE_CACHE.get(p)
    if v is not None:
        return v
    a, sr = _load_audio(p)
    fp = _single_cache_helper.fingerprint(a, sr)
    _FP_SINGLE_CACHE[p] = fp.vec
    return fp.vec


def _fp_twostream(p: Path) -> tuple[np.ndarray, np.ndarray]:
    v = _FP_TWOSTREAM_CACHE.get(p)
    if v is not None:
        return v
    a, sr = _load_audio(p)
    fp = _twostream_cache_helper.fingerprint(a, sr)
    _FP_TWOSTREAM_CACHE[p] = (fp.tone, fp.segment)
    return _FP_TWOSTREAM_CACHE[p]


def _score_single(p1: Path, p2: Path) -> float:
    v1 = _fp_single(p1); v2 = _fp_single(p2)
    return float(np.dot(v1, v2))


def _score_twostream(p1: Path, p2: Path) -> tuple[float, float]:
    t1, s1 = _fp_twostream(p1); t2, s2 = _fp_twostream(p2)
    return float(np.dot(t1, t2)), float(np.dot(s1, s2))


# ---------------- Sweep ----------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def sweep_single(pos_scores: list[float], neg_scores: list[float]) -> dict:
    taus = np.round(np.arange(0.70, 0.99 + 1e-9, 0.02), 4).tolist()
    pos = np.asarray(pos_scores); neg = np.asarray(neg_scores)
    rows = []
    for tau in taus:
        tp = int((pos >= tau).sum())
        fn = int((pos <  tau).sum())
        fp = int((neg >= tau).sum())
        tn = int((neg <  tau).sum())
        p, r, f = _prf(tp, fp, fn)
        rows.append({"tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    best = max(rows, key=lambda x: x["f1"])
    # operating point — pick the row whose tau is closest to current
    op = min(rows, key=lambda x: abs(x["tau"] - CURRENT_SINGLE_TAU))
    return {"sweep": rows, "best_f1": best, "operating_point": op,
            "current_tau": CURRENT_SINGLE_TAU,
            "score_stats": {
                "pos_mean": float(pos.mean()), "pos_std": float(pos.std()),
                "neg_mean": float(neg.mean()), "neg_std": float(neg.std()),
                "pos_min":  float(pos.min()),  "pos_max": float(pos.max()),
                "neg_min":  float(neg.min()),  "neg_max": float(neg.max()),
            }}


def sweep_twostream(pos_scores: list[tuple[float, float]],
                    neg_scores: list[tuple[float, float]]) -> dict:
    tone_taus = [0.70, 0.85, 0.95]
    seg_taus  = [0.85, 0.93, 0.97]
    pos_t = np.asarray([t for t, _ in pos_scores])
    pos_s = np.asarray([s for _, s in pos_scores])
    neg_t = np.asarray([t for t, _ in neg_scores])
    neg_s = np.asarray([s for _, s in neg_scores])
    rows = []
    for tt in tone_taus:
        for st in seg_taus:
            pos_hit = (pos_t >= tt) & (pos_s >= st)
            neg_hit = (neg_t >= tt) & (neg_s >= st)
            tp = int(pos_hit.sum()); fn = int((~pos_hit).sum())
            fp = int(neg_hit.sum()); tn = int((~neg_hit).sum())
            p, r, f = _prf(tp, fp, fn)
            rows.append({"tau_tone": tt, "tau_seg": st,
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    best = max(rows, key=lambda x: x["f1"])
    op = min(rows, key=lambda x: abs(x["tau_tone"] - CURRENT_TWOSTREAM_TONE)
                                  + abs(x["tau_seg"]  - CURRENT_TWOSTREAM_SEG))
    return {"sweep": rows, "best_f1": best, "operating_point": op,
            "current_tau_tone": CURRENT_TWOSTREAM_TONE,
            "current_tau_seg":  CURRENT_TWOSTREAM_SEG,
            "score_stats": {
                "pos_tone_mean": float(pos_t.mean()), "pos_tone_std": float(pos_t.std()),
                "pos_seg_mean":  float(pos_s.mean()), "pos_seg_std":  float(pos_s.std()),
                "neg_tone_mean": float(neg_t.mean()), "neg_tone_std": float(neg_t.std()),
                "neg_seg_mean":  float(neg_s.mean()), "neg_seg_std":  float(neg_s.std()),
            }}


# ---------------- Pretty print ----------------

def _classify_op(op: dict) -> str:
    p, r = op["precision"], op["recall"]
    if p >= 0.95 and r < 0.80:
        return "CONSERVATIVE (high precision, low recall — refuses to hit unless very sure)"
    if r >= 0.95 and p < 0.80:
        return "AGGRESSIVE (high recall, low precision — hits often, risks wrong translations)"
    if p >= 0.90 and r >= 0.90:
        return "BALANCED (both high)"
    if p < 0.70 and r < 0.70:
        return "POOR (both low — threshold poorly placed)"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",  default="Results/threshold_sweep.json")
    args = ap.parse_args()

    print("[data] sampling labelled pairs from VIVOS train ...")
    pos_pairs, neg_pairs, meta = _build_pairs(
        target_per_class=args.target_per_class, seed=args.seed)
    print(f"  unique texts in VIVOS train     : {meta['n_unique_texts']}")
    print(f"  multi-speaker text groups       : {meta['n_multi_speaker_groups']}")
    print(f"  positive pairs (same-text diff-spk): {len(pos_pairs)}")
    print(f"  negative pairs (diff-text)         : {len(neg_pairs)}")
    if len(pos_pairs) < 100 or len(neg_pairs) < 100:
        print("  [WARN] fewer than 100 pairs in at least one class.")

    print("\n[score] computing single-vector cosines ...")
    t0 = time.perf_counter()
    pos_single = [_score_single(a, b) for a, b in pos_pairs]
    neg_single = [_score_single(a, b) for a, b in neg_pairs]
    print(f"  {len(pos_single)+len(neg_single)} cosines in {time.perf_counter()-t0:.1f}s")

    print("[score] computing two-stream (tone, segment) cosines ...")
    t0 = time.perf_counter()
    pos_two = [_score_twostream(a, b) for a, b in pos_pairs]
    neg_two = [_score_twostream(a, b) for a, b in neg_pairs]
    print(f"  {len(pos_two)+len(neg_two)} pairs in {time.perf_counter()-t0:.1f}s")

    single = sweep_single(pos_single, neg_single)
    twostream = sweep_twostream(pos_two, neg_two)

    out = {
        "meta":      meta,
        "single":    single,
        "twostream": twostream,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Console report ----------------
    print("\n" + "=" * 78)
    print("SINGLE-VECTOR UtteranceCache — τ sweep")
    print("=" * 78)
    print(f"{'tau':>6} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    for row in single["sweep"]:
        print(f"{row['tau']:>6.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
              f"{row['f1']:>8.4f} {row['tp']:>5} {row['fp']:>5} {row['fn']:>5} {row['tn']:>5}")
    b = single["best_f1"]; o = single["operating_point"]; s = single["score_stats"]
    print(f"\n  pos cos: mean={s['pos_mean']:.3f} std={s['pos_std']:.3f}  "
          f"[min={s['pos_min']:.3f} max={s['pos_max']:.3f}]")
    print(f"  neg cos: mean={s['neg_mean']:.3f} std={s['neg_std']:.3f}  "
          f"[min={s['neg_min']:.3f} max={s['neg_max']:.3f}]")
    print(f"  BEST F1 @ τ={b['tau']:.2f}: P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}")
    print(f"  CURRENT τ={CURRENT_SINGLE_TAU} (closest row τ={o['tau']:.2f}): "
          f"P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    print(f"  → {_classify_op(o)}")

    print("\n" + "=" * 78)
    print("TWO-STREAM TwoStreamCache — (τ_tone × τ_seg) grid")
    print("=" * 78)
    print(f"{'tone':>6} {'seg':>6} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    for row in twostream["sweep"]:
        print(f"{row['tau_tone']:>6.2f} {row['tau_seg']:>6.2f} {row['precision']:>8.4f} "
              f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['tp']:>5} {row['fp']:>5} "
              f"{row['fn']:>5} {row['tn']:>5}")
    b = twostream["best_f1"]; o = twostream["operating_point"]
    print(f"\n  BEST F1 @ τ_tone={b['tau_tone']:.2f} τ_seg={b['tau_seg']:.2f}: "
          f"P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}")
    print(f"  CURRENT τ_tone={CURRENT_TWOSTREAM_TONE} τ_seg={CURRENT_TWOSTREAM_SEG} "
          f"(closest row {o['tau_tone']:.2f}/{o['tau_seg']:.2f}): "
          f"P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
    print(f"  → {_classify_op(o)}")

    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
