"""
2D threshold-sensitivity sweep over (tau_A, tau_B) for the two-stream voting
cache (ACL reviewer Q9).

Reviewer concern: how sensitive are tau_A and tau_B to noise and speakers?
Provide threshold-sensitivity curves and an automatic calibration procedure
for new domains.

Protocol:
  1. Reuse the labelled pair set from prototype/threshold_sweep.py:
       POSITIVE = same prompt, different VIVOS speakers (200 pairs)
       NEGATIVE = different prompts (200 pairs)
     Plus the tonal-minimal-pair clips (5 syllables × 6 tones, 30 clips)
     for a separate false-hit-rate measurement against tonal variants.
  2. Sweep tau_A ∈ {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}
     × tau_B ∈ {0.85, 0.88, 0.90, 0.92, 0.93, 0.95}  → 6×6 = 36 cells.
  3. At each cell: precision, recall, F1, tonal-false-hit-rate.
  4. Save Results/threshold_2d_sweep.json (full grid).
  5. Plot a 2D F1 heatmap → papers/figures/fig_threshold_heatmap.png (200 dpi).
  6. Auto-calibration recipe (function + ≤80-word description):
        Given an unlabeled calibration set of audio clips with text labels
        (speaker IDs available), construct same-text-diff-speaker pos pairs
        and diff-text neg pairs, then pick (tau_A, tau_B) ∈ GRID maximizing
        F1 with a precision floor of 0.85 to keep the deployment safety
        asymmetry.
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

from two_stream_voting import TwoStreamCache                        # noqa: E402


PROJECT     = HERE.parent
VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

TAU_A_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_B_GRID = [0.85, 0.88, 0.90, 0.92, 0.93, 0.95]

DEFAULT_TAU_A = 0.85
DEFAULT_TAU_B = 0.93


# ---------------- VIVOS pair sampling (same protocol as threshold_sweep.py) ----------------

def _load_vivos_text_groups() -> dict[str, list[tuple[Path, str]]]:
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


def _build_pairs(target_per_class: int = 200, seed: int = 42):
    groups = _load_vivos_text_groups()
    rng = np.random.default_rng(seed)

    pos_pairs: list[tuple[Path, Path]] = []
    for text, entries in groups.items():
        spk_to_paths: dict[str, list[Path]] = defaultdict(list)
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
                pos_pairs.append((p1, p2))
                if len(pos_pairs) >= target_per_class:
                    break
            if len(pos_pairs) >= target_per_class:
                break
        if len(pos_pairs) >= target_per_class:
            break

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

    return pos_pairs, neg_pairs


# ---------------- Tonal-minimal-pair clips ----------------

def _find_minpair_clips() -> list[Path]:
    """Try the canonical minimal-pair location used by minimal_pair_eval.py."""
    candidates = [
        PROJECT / "data" / "raw" / "tonal_minimal_pairs",
        PROJECT / "data" / "processed" / "tonal_minimal_pairs",
        PROJECT / "data" / "tonal_minimal_pairs",
        HERE / "tonal_minimal_pairs",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            wavs = sorted(c.rglob("*.wav"))
            if wavs:
                return wavs
    return []


def _synth_minpair_audios_inmemory() -> list[tuple[str, np.ndarray, int]]:
    """Synthesize the 36 minimal-pair clips via MMS-TTS in memory.
    Returns list of (key, audio, sr) tuples where key = '<syl>_<variant>'."""
    try:
        from minimal_pair_eval import MINIMAL_PAIR_SYLLABLES, CARRIER, synth_audio
        from transformers import AutoTokenizer, VitsModel
    except Exception as e:
        print(f"  [warn] MMS-TTS unavailable: {e}", flush=True)
        return []
    print("  [synth] loading MMS-TTS-vie ...", flush=True)
    tok   = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = int(model.config.sampling_rate)
    out: list[tuple[str, np.ndarray, int]] = []
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            text = CARRIER.format(syl=v)
            try:
                a = synth_audio(text, tok, model, sr)
                out.append((f"{base}_{v}", a, sr))
            except Exception as e:
                print(f"  [warn] synth fail for {text!r}: {e}", flush=True)
    return out


def _minpair_pairs(clips: list[Path]) -> list[tuple[Path, Path, str]]:
    """Build all same-syllable cross-tone pairs.
    Expects filenames like 'ma_tone1.wav' or 'ma_huyen.wav'; syllable = stem
    up to first '_'. Returns (a, b, syllable) tuples where syllable matches
    but the tone (rest of stem) differs."""
    by_syl: dict[str, list[Path]] = defaultdict(list)
    for p in clips:
        stem = p.stem
        syl = stem.split("_")[0] if "_" in stem else stem
        by_syl[syl].append(p)
    out: list[tuple[Path, Path, str]] = []
    for syl, ps in by_syl.items():
        if len(ps) < 2:
            continue
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                out.append((ps[i], ps[j], syl))
    return out


# ---------------- Fingerprint helpers ----------------

_FP_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
_helper = TwoStreamCache()


def _load_audio(p: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(p), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), int(sr)


def _fp(p: Path) -> tuple[np.ndarray, np.ndarray]:
    v = _FP_CACHE.get(p)
    if v is not None:
        return v
    a, sr = _load_audio(p)
    fp = _helper.fingerprint(a, sr)
    _FP_CACHE[p] = (fp.tone, fp.segment)
    return _FP_CACHE[p]


def _score(p1: Path, p2: Path) -> tuple[float, float]:
    t1, s1 = _fp(p1); t2, s2 = _fp(p2)
    return float(np.dot(t1, t2)), float(np.dot(s1, s2))


# ---------------- Grid sweep ----------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def grid_sweep(pos_scores: list[tuple[float, float]],
               neg_scores: list[tuple[float, float]],
               minpair_scores: list[tuple[float, float]] | None,
               tau_a_grid: list[float] = TAU_A_GRID,
               tau_b_grid: list[float] = TAU_B_GRID) -> dict:
    pos_t = np.asarray([t for t, _ in pos_scores])
    pos_s = np.asarray([s for _, s in pos_scores])
    neg_t = np.asarray([t for t, _ in neg_scores])
    neg_s = np.asarray([s for _, s in neg_scores])
    if minpair_scores:
        mp_t = np.asarray([t for t, _ in minpair_scores])
        mp_s = np.asarray([s for _, s in minpair_scores])
    else:
        mp_t = mp_s = np.asarray([])

    rows: list[dict] = []
    for ta in tau_a_grid:
        for tb in tau_b_grid:
            pos_hit = (pos_t >= ta) & (pos_s >= tb)
            neg_hit = (neg_t >= ta) & (neg_s >= tb)
            tp = int(pos_hit.sum()); fn = int((~pos_hit).sum())
            fp_ = int(neg_hit.sum()); tn = int((~neg_hit).sum())
            p, r, f = _prf(tp, fp_, fn)
            if mp_t.size:
                mp_hit  = (mp_t >= ta) & (mp_s >= tb)
                mp_fhr  = float(mp_hit.mean())
                mp_n    = int(mp_t.size)
                mp_fhits = int(mp_hit.sum())
            else:
                mp_fhr = -1.0
                mp_n = 0
                mp_fhits = 0
            rows.append({
                "tau_A": ta, "tau_B": tb,
                "tp": tp, "fp": fp_, "fn": fn, "tn": tn,
                "precision": round(p, 4),
                "recall":    round(r, 4),
                "f1":        round(f, 4),
                "tonal_fhr":     round(mp_fhr, 4) if mp_fhr >= 0 else None,
                "tonal_n":       mp_n,
                "tonal_fhits":   mp_fhits,
            })
    return {
        "tau_A_grid": tau_a_grid,
        "tau_B_grid": tau_b_grid,
        "grid": rows,
        "score_stats": {
            "pos_tone_mean": float(pos_t.mean()), "pos_tone_std": float(pos_t.std()),
            "pos_seg_mean":  float(pos_s.mean()), "pos_seg_std":  float(pos_s.std()),
            "neg_tone_mean": float(neg_t.mean()), "neg_tone_std": float(neg_t.std()),
            "neg_seg_mean":  float(neg_s.mean()), "neg_seg_std":  float(neg_s.std()),
        },
    }


# ---------------- Auto-calibration recipe ----------------

def auto_calibrate(pos_scores: list[tuple[float, float]],
                   neg_scores: list[tuple[float, float]],
                   tau_a_grid: list[float] = TAU_A_GRID,
                   tau_b_grid: list[float] = TAU_B_GRID,
                   precision_floor: float = 0.85) -> dict:
    """Auto-calibration: given a calibration set with positive/negative pairs,
    sweep the (tau_A, tau_B) grid and return the cell maximizing F1 subject
    to precision >= precision_floor. The precision floor enforces the
    deployment safety asymmetry (a false hit shows a wrong translation;
    a miss only falls back to the cascade). If no cell meets the floor,
    return the cell with maximum precision among the top-F1 quartile.
    """
    out = grid_sweep(pos_scores, neg_scores, None, tau_a_grid, tau_b_grid)
    rows = out["grid"]
    safe = [r for r in rows if r["precision"] >= precision_floor]
    if safe:
        best = max(safe, key=lambda r: (r["f1"], r["precision"]))
        rule = f"max-F1 subject to P>={precision_floor}"
    else:
        rows_sorted = sorted(rows, key=lambda r: -r["f1"])
        top_quartile = rows_sorted[: max(1, len(rows_sorted) // 4)]
        best = max(top_quartile, key=lambda r: r["precision"])
        rule = (f"no cell meets P>={precision_floor}; "
                f"chose max-P among top-F1 quartile")
    return {"selected": best, "rule": rule, "precision_floor": precision_floor}


# ---------------- Plotting ----------------

def plot_heatmap(grid_result: dict, out_path: Path, dpi: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau_a = grid_result["tau_A_grid"]
    tau_b = grid_result["tau_B_grid"]
    n_a = len(tau_a); n_b = len(tau_b)
    F1 = np.zeros((n_b, n_a), dtype=np.float32)
    P  = np.zeros_like(F1)
    R  = np.zeros_like(F1)
    for r in grid_result["grid"]:
        ia = tau_a.index(r["tau_A"])
        ib = tau_b.index(r["tau_B"])
        F1[ib, ia] = r["f1"]
        P[ib, ia]  = r["precision"]
        R[ib, ia]  = r["recall"]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(F1, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=max(0.3, float(F1.max())))
    ax.set_xticks(range(n_a))
    ax.set_xticklabels([f"{x:.2f}" for x in tau_a])
    ax.set_yticks(range(n_b))
    ax.set_yticklabels([f"{x:.2f}" for x in tau_b])
    ax.set_xlabel(r"$\tau_A$ (tone-stream threshold)")
    ax.set_ylabel(r"$\tau_B$ (segmental-stream threshold)")
    ax.set_title("Two-stream cache F1 over the threshold grid")

    for ib in range(n_b):
        for ia in range(n_a):
            v = F1[ib, ia]
            color = "white" if v < (float(F1.max()) * 0.55) else "black"
            ax.text(ia, ib, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    # Mark default operating point.
    try:
        ia = tau_a.index(DEFAULT_TAU_A)
        ib = tau_b.index(DEFAULT_TAU_B)
        ax.scatter([ia], [ib], s=180, marker="o",
                   facecolors="none", edgecolors="red", linewidths=2.0,
                   label=fr"default ($\tau_A$={DEFAULT_TAU_A}, $\tau_B$={DEFAULT_TAU_B})")
        ax.legend(loc="lower left", fontsize=8)
    except ValueError:
        pass

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="Results/threshold_2d_sweep.json")
    ap.add_argument("--fig",
                    default="papers/figures/fig_threshold_heatmap.png")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("[data] sampling labelled VIVOS pairs ...", flush=True)
    pos_pairs, neg_pairs = _build_pairs(
        target_per_class=args.target_per_class, seed=args.seed)
    print(f"  positive pairs: {len(pos_pairs)} | negative pairs: {len(neg_pairs)}",
          flush=True)

    print("[data] scanning tonal-minimal-pair clips ...", flush=True)
    mp_clips = _find_minpair_clips()
    mp_inmem: list[tuple[str, np.ndarray, int]] = []
    if mp_clips:
        mp_pairs = _minpair_pairs(mp_clips)
        print(f"  found {len(mp_clips)} clips on disk → {len(mp_pairs)} "
              f"same-syllable cross-tone pairs", flush=True)
        minpair_scores = [_score(a, b) for a, b, _ in mp_pairs]
    else:
        print("  no minimal-pair clips on disk; synthesizing via MMS-TTS-vie",
              flush=True)
        mp_inmem = _synth_minpair_audios_inmemory()
        if not mp_inmem:
            print("  (synthesis unavailable; tonal-FHR will be omitted)",
                  flush=True)
            mp_pairs = []
            minpair_scores = []
        else:
            # Build same-syllable cross-tone pairs and score in-memory.
            from collections import defaultdict as _dd
            by_syl: dict[str, list[tuple[str, np.ndarray, int]]] = _dd(list)
            for key, a, sr in mp_inmem:
                syl = key.split("_", 1)[0]
                by_syl[syl].append((key, a, sr))
            mp_pairs_keys: list[tuple[str, str]] = []
            mp_fps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for key, a, sr in mp_inmem:
                fp = _helper.fingerprint(a, sr)
                mp_fps[key] = (fp.tone, fp.segment)
            for syl, items in by_syl.items():
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        mp_pairs_keys.append((items[i][0], items[j][0]))
            minpair_scores = []
            for k1, k2 in mp_pairs_keys:
                t1, s1 = mp_fps[k1]; t2, s2 = mp_fps[k2]
                minpair_scores.append(
                    (float(np.dot(t1, t2)), float(np.dot(s1, s2))))
            mp_pairs = mp_pairs_keys
            mp_clips = list(mp_fps.keys())
            print(f"  synthesized {len(mp_inmem)} clips → {len(mp_pairs)} "
                  f"same-syllable cross-tone pairs", flush=True)

    print("[score] two-stream cosines on VIVOS pairs ...", flush=True)
    t0 = time.perf_counter()
    pos_scores = [_score(a, b) for a, b in pos_pairs]
    neg_scores = [_score(a, b) for a, b in neg_pairs]
    print(f"  {len(pos_scores) + len(neg_scores)} pairs in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    grid = grid_sweep(pos_scores, neg_scores, minpair_scores or None,
                      TAU_A_GRID, TAU_B_GRID)
    cal  = auto_calibrate(pos_scores, neg_scores, TAU_A_GRID, TAU_B_GRID,
                          precision_floor=0.85)

    default_row = next((r for r in grid["grid"]
                        if r["tau_A"] == DEFAULT_TAU_A
                        and r["tau_B"] == DEFAULT_TAU_B), None)
    best_f1     = max(grid["grid"], key=lambda r: r["f1"])

    out = {
        "meta": {
            "n_pos_pairs": len(pos_pairs),
            "n_neg_pairs": len(neg_pairs),
            "n_minpair_clips": len(mp_clips),
            "n_minpair_pairs": len(mp_pairs),
            "tau_A_grid": TAU_A_GRID,
            "tau_B_grid": TAU_B_GRID,
            "seed": args.seed,
        },
        "grid":             grid,
        "default_operating_point": default_row,
        "best_f1":          best_f1,
        "auto_calibration": cal,
    }

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] wrote {out_path}")

    fig_path = PROJECT / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmap(grid, fig_path, dpi=200)
    print(f"[done] wrote {fig_path}")

    # Console summary.
    print("\n" + "=" * 78)
    print(f"2D THRESHOLD SWEEP — {len(TAU_A_GRID)} x {len(TAU_B_GRID)} = "
          f"{len(TAU_A_GRID) * len(TAU_B_GRID)} cells")
    print("=" * 78)
    print(f"{'tau_A':>6} {'tau_B':>6} {'P':>8} {'R':>8} {'F1':>8} "
          f"{'TP':>4} {'FP':>4} {'tonFHR':>7}")
    for r in grid["grid"]:
        tFHR = "n/a" if r["tonal_fhr"] is None else f"{r['tonal_fhr']:.3f}"
        print(f"{r['tau_A']:>6.2f} {r['tau_B']:>6.2f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
              f"{r['tp']:>4} {r['fp']:>4} {tFHR:>7}")

    print("\n  default (τ_A=0.85, τ_B=0.93): "
          f"P={default_row['precision']:.4f} R={default_row['recall']:.4f} "
          f"F1={default_row['f1']:.4f}")
    print(f"  best F1 cell @ (τ_A={best_f1['tau_A']:.2f}, "
          f"τ_B={best_f1['tau_B']:.2f}): "
          f"P={best_f1['precision']:.4f} R={best_f1['recall']:.4f} "
          f"F1={best_f1['f1']:.4f}")
    sel = cal["selected"]
    print(f"  auto-calib  ({cal['rule']}): "
          f"(τ_A={sel['tau_A']:.2f}, τ_B={sel['tau_B']:.2f}) "
          f"P={sel['precision']:.4f} R={sel['recall']:.4f} "
          f"F1={sel['f1']:.4f}")

    # Robustness band: report F1 range over the central rectangle the paper
    # will claim is robust.
    central = [r for r in grid["grid"]
               if 0.80 <= r["tau_A"] <= 0.90 and 0.90 <= r["tau_B"] <= 0.93]
    if central:
        f1s = [r["f1"] for r in central]
        print(f"  central robust band (τ_A∈[0.80,0.90] × τ_B∈[0.90,0.93]): "
              f"F1 min={min(f1s):.4f} max={max(f1s):.4f} "
              f"(n={len(central)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
