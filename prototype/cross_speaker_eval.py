"""
Cross-speaker generalization of the two-stream cache (reviewer Q2).

Split the 46 VIVOS-train speakers into a 30-speaker SEEN group (calibration)
and a 16-speaker UNSEEN group (test). For each group, build positive pairs
(same prompt, two different speakers from the group) and negative pairs
(same speaker, two different prompts). Calibrate the tone-stream threshold
tau_A on SEEN under the existence rule with tau_B fixed at 0.93, then
evaluate the SAME tau on UNSEEN to quantify the calibration generalization
gap.

Outputs:
  Results/cross_speaker_eval.json

Run from project root:
    python prototype/cross_speaker_eval.py
"""
from __future__ import annotations

import json
import os
import sys
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

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from two_stream_voting import TwoStreamCache                    # noqa: E402

VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

SEED          = 42
N_SEEN        = 30
N_UNSEEN      = 16
N_POS_TARGET  = 200
N_NEG_TARGET  = 200
TAU_B         = 0.93
TAU_A_GRID    = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85,
                 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]


def _load_audio(p: Path):
    a, sr = sf.read(str(p), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), int(sr)


def _load_groups():
    """Return text -> list[(wav_path, speaker_id)] from VIVOS train."""
    prompts_file = VIVOS_TRAIN / "prompts.txt"
    text_by_id = {}
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        uid, _, text = line.partition(" ")
        text_by_id[uid.strip()] = text.strip().lower()
    text_groups = defaultdict(list)
    waves_root = VIVOS_TRAIN / "waves"
    for spk_dir in sorted(waves_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            text = text_by_id.get(wav.stem)
            if text is None:
                continue
            text_groups[text].append((wav, spk))
    return text_groups


def _build_positive_pairs(text_groups, speaker_set, target, seed_offset=0):
    """Same-text different-speaker pairs, both speakers in speaker_set."""
    rng = np.random.default_rng(SEED + seed_offset)
    pairs = []
    texts = list(text_groups.keys())
    rng.shuffle(texts)
    for text in texts:
        entries = [(p, s) for p, s in text_groups[text] if s in speaker_set]
        spk_to_paths = defaultdict(list)
        for p, s in entries:
            spk_to_paths[s].append(p)
        if len(spk_to_paths) < 2:
            continue
        spks = list(spk_to_paths.keys())
        rng.shuffle(spks)
        for i in range(len(spks) - 1):
            for j in range(i + 1, len(spks)):
                p1 = spk_to_paths[spks[i]][0]
                p2 = spk_to_paths[spks[j]][0]
                pairs.append((p1, p2))
                if len(pairs) >= target: return pairs
    return pairs


def _build_negative_pairs(text_groups, speaker_set, target, seed_offset=1):
    """Same-speaker different-text pairs (speaker in speaker_set)."""
    rng = np.random.default_rng(SEED + seed_offset)
    spk_to_texts = defaultdict(list)
    for text, entries in text_groups.items():
        for p, s in entries:
            if s in speaker_set:
                spk_to_texts[s].append((p, text))
    pairs = []
    spks = list(spk_to_texts.keys())
    rng.shuffle(spks)
    for spk in spks:
        items = spk_to_texts[spk]
        if len(items) < 2:
            continue
        rng.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            p1, t1 = items[i]
            p2, t2 = items[i + 1]
            if t1 == t2:
                continue
            pairs.append((p1, p2))
            if len(pairs) >= target: return pairs
    return pairs


def _existence_score(fp_a, fp_b, tau_A, tau_B):
    t = float(np.dot(fp_a.tone, fp_b.tone))
    s = float(np.dot(fp_a.segment, fp_b.segment))
    return (t >= tau_A) and (s >= tau_B), t, s


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def _eval(fp_pos_pairs, fp_neg_pairs, tau_A, tau_B):
    tp = sum(1 for a, b in fp_pos_pairs if _existence_score(a, b, tau_A, tau_B)[0])
    fp_ = sum(1 for a, b in fp_neg_pairs if _existence_score(a, b, tau_A, tau_B)[0])
    fn = len(fp_pos_pairs) - tp
    tn = len(fp_neg_pairs) - fp_
    p, r, f1 = _prf(tp, fp_, fn)
    return {
        "tau_A": round(tau_A, 4),
        "tau_B": round(tau_B, 4),
        "tp": tp, "fp": fp_, "fn": fn, "tn": tn,
        "precision": p, "recall": r, "f1": f1,
    }


def main() -> int:
    print("[load] VIVOS train metadata ...")
    text_groups = _load_groups()
    all_speakers = sorted({s for entries in text_groups.values() for _, s in entries})
    print(f"  speakers: {len(all_speakers)}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_speakers))
    seen_spks   = set(all_speakers[i] for i in perm[:N_SEEN])
    unseen_spks = set(all_speakers[i] for i in perm[N_SEEN:N_SEEN + N_UNSEEN])
    print(f"  seen  : {sorted(seen_spks)}")
    print(f"  unseen: {sorted(unseen_spks)}")

    print(f"[pairs] building positive/negative pairs (target {N_POS_TARGET}/{N_NEG_TARGET}) ...")
    seen_pos = _build_positive_pairs(text_groups, seen_spks,   N_POS_TARGET, seed_offset=10)
    seen_neg = _build_negative_pairs(text_groups, seen_spks,   N_NEG_TARGET, seed_offset=11)
    unseen_pos = _build_positive_pairs(text_groups, unseen_spks, N_POS_TARGET, seed_offset=20)
    unseen_neg = _build_negative_pairs(text_groups, unseen_spks, N_NEG_TARGET, seed_offset=21)
    print(f"  seen   pos={len(seen_pos)}  neg={len(seen_neg)}")
    print(f"  unseen pos={len(unseen_pos)}  neg={len(unseen_neg)}")

    print("[fp] computing two-stream fingerprints on all unique wavs ...")
    fp_helper = TwoStreamCache()
    fp_cache: dict = {}
    def _get_fp(p: Path):
        if p in fp_cache: return fp_cache[p]
        a, asr = _load_audio(p)
        fp = fp_helper.fingerprint(a, asr)
        fp_cache[p] = fp
        return fp

    unique_wavs = set()
    for src in (seen_pos, seen_neg, unseen_pos, unseen_neg):
        for a, b in src:
            unique_wavs.add(a); unique_wavs.add(b)
    print(f"  {len(unique_wavs)} unique wavs")
    for i, p in enumerate(sorted(unique_wavs, key=lambda x: str(x))):
        _get_fp(p)
        if (i + 1) % 100 == 0:
            print(f"  fp {i+1}/{len(unique_wavs)}")

    seen_pos_fp   = [(_get_fp(a), _get_fp(b)) for a, b in seen_pos]
    seen_neg_fp   = [(_get_fp(a), _get_fp(b)) for a, b in seen_neg]
    unseen_pos_fp = [(_get_fp(a), _get_fp(b)) for a, b in unseen_pos]
    unseen_neg_fp = [(_get_fp(a), _get_fp(b)) for a, b in unseen_neg]

    print(f"\n[calibrate] sweep tau_A on SEEN; pick argmax F1 ...")
    seen_sweep = []
    for tau_A in TAU_A_GRID:
        r = _eval(seen_pos_fp, seen_neg_fp, tau_A, TAU_B)
        seen_sweep.append(r)
        print(f"  tau_A={tau_A:.2f}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")

    best_seen = max(seen_sweep, key=lambda r: r["f1"])
    tau_A_cal = best_seen["tau_A"]
    print(f"  -> calibrated tau_A = {tau_A_cal} (F1_seen = {best_seen['f1']})")

    print(f"\n[transfer] evaluate SAME tau_A={tau_A_cal} on UNSEEN ...")
    unseen_at_cal = _eval(unseen_pos_fp, unseen_neg_fp, tau_A_cal, TAU_B)
    print(f"  UNSEEN @ cal tau_A: P={unseen_at_cal['precision']:.3f} R={unseen_at_cal['recall']:.3f} F1={unseen_at_cal['f1']:.3f}")
    print(f"  SEEN   @ cal tau_A: P={best_seen['precision']:.3f} R={best_seen['recall']:.3f} F1={best_seen['f1']:.3f}")
    delta_f1 = round(unseen_at_cal["f1"] - best_seen["f1"], 4)
    print(f"  delta F1 (unseen - seen) = {delta_f1:+.4f}")

    print(f"\n[recalibrate-unseen] sweep tau_A on UNSEEN to find F1-matching point ...")
    unseen_sweep = []
    for tau_A in TAU_A_GRID:
        r = _eval(unseen_pos_fp, unseen_neg_fp, tau_A, TAU_B)
        unseen_sweep.append(r)

    best_unseen = max(unseen_sweep, key=lambda r: r["f1"])
    tau_A_match = None
    for r in unseen_sweep:
        if r["f1"] >= best_seen["f1"]:
            tau_A_match = r["tau_A"]
            break
    if tau_A_match is None:
        tau_A_match_note = (f"no tau_A in grid lifts UNSEEN F1 to SEEN's best ({best_seen['f1']}); "
                            f"UNSEEN max F1 = {best_unseen['f1']} at tau_A={best_unseen['tau_A']}")
    else:
        tau_A_match_note = (f"UNSEEN matches SEEN F1 at tau_A={tau_A_match} "
                            f"(calibration gap = {tau_A_match - tau_A_cal:+.2f})")
    print(f"  {tau_A_match_note}")

    headline = (f"Cross-speaker F1: seen {best_seen['f1']} at tau_A={tau_A_cal}, "
                f"unseen {unseen_at_cal['f1']} at the SAME tau, delta {delta_f1:+.4f}. "
                f"{tau_A_match_note}.")

    out = {
        "seed":            SEED,
        "n_seen_speakers":   len(seen_spks),
        "n_unseen_speakers": len(unseen_spks),
        "seen_speakers":   sorted(seen_spks),
        "unseen_speakers": sorted(unseen_spks),
        "tau_B":           TAU_B,
        "tau_A_grid":      TAU_A_GRID,
        "n_pairs": {
            "seen_pos":   len(seen_pos),  "seen_neg":   len(seen_neg),
            "unseen_pos": len(unseen_pos), "unseen_neg": len(unseen_neg),
        },
        "seen_sweep":         seen_sweep,
        "unseen_sweep":       unseen_sweep,
        "calibrated_tau_A":   tau_A_cal,
        "seen_at_calibrated":   best_seen,
        "unseen_at_calibrated": unseen_at_cal,
        "unseen_best":          best_unseen,
        "tau_A_match_unseen":   tau_A_match,
        "delta_f1_unseen_minus_seen": delta_f1,
        "headline": headline,
    }
    out_path = PROJECT / "Results" / "cross_speaker_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    print(f"[headline] {headline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
