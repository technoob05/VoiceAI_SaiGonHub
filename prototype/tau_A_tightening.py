"""
tau_A tightening sweep for the two-stream existence rule (reviewer Q1 follow-up).

The existence rule at (tau_A=0.85, tau_B=0.93) drops correct-miss from 92%
(argmax-equality) to 72%. This script sweeps tau_A in {0.85..0.93 step 0.01}
with tau_B fixed at 0.93 to find the operating point that recovers tonal
safety while preserving recall on a 200-pair VIVOS positive set.

Outputs:
  Results/tau_A_tightening.json
  papers/figures/fig_tau_a_tightening.png

Pure NumPy + matplotlib; CPU only.

Run from project root:
    python prototype/tau_A_tightening.py
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

from two_stream_voting import (                                  # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
    MINIMAL_PAIR_SYLLABLES,
    CARRIER,
    _synth,
)


VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

TAU_A_GRID = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]
TAU_B = 0.93

SEED = 42
N_VIVOS_POS_PAIRS = 200

# safety acceptance criteria for "recommended" tau_A
MIN_CORRECT_MISS  = 24   # >= 24/25
MIN_VIVOS_RECALL  = 0.50 # at least half of positive pairs still hit

# ---------------- VIVOS positive-pair sampling (same protocol as threshold_2d_sweep.py) ----------------

def _load_vivos_text_groups() -> dict:
    prompts_file = VIVOS_TRAIN / "prompts.txt"
    text_by_id = {}
    for line in prompts_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        text_by_id[utt_id.strip()] = text.strip().lower()
    groups = defaultdict(list)
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


def _build_vivos_positive_pairs(target: int = 200, seed: int = SEED):
    groups = _load_vivos_text_groups()
    rng = np.random.default_rng(seed)
    pos = []
    for text, entries in groups.items():
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
                pos.append((p1, p2))
                if len(pos) >= target: break
            if len(pos) >= target: break
        if len(pos) >= target: break
    return pos


def _load_audio(p: Path):
    a, sr = sf.read(str(p), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a.astype(np.float32), int(sr)


# ---------------- Evaluation core ----------------

def _eval_existence_rule(audio_by_key, sr, vivos_fp_pairs, tau_A: float, tau_B: float) -> dict:
    """Run the same exp1 (self-lookup) + exp2 (correct-miss) protocol used by
    two_stream_existence_eval.py at (tau_A, tau_B), plus VIVOS positive-pair
    recall under the existence rule.

    vivos_fp_pairs: list of (fp_a, fp_b) TwoStreamFingerprint tuples (precomputed).
    """
    cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B)

    # exp1: store all variants, query each
    cache_full = TwoStreamCache(cfg=cfg)
    for k, a in audio_by_key.items():
        fp = cache_full.fingerprint(a, sr)
        cache_full.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})
    self_hits = 0
    for k, a in audio_by_key.items():
        fp = cache_full.fingerprint(a, sr)
        h = cache_full.lookup_existence(fp, "en")
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1
    self_total = len(audio_by_key)

    # exp2: store only the ngang variant per syllable, probe other tones
    cache_ngang = TwoStreamCache(cfg=cfg)
    for base in MINIMAL_PAIR_SYLLABLES:
        ma_key = (base, MINIMAL_PAIR_SYLLABLES[base][0])
        if ma_key in audio_by_key:
            fp = cache_ngang.fingerprint(audio_by_key[ma_key], sr)
            cache_ngang.put(fp, "en", {
                "vi":  MINIMAL_PAIR_SYLLABLES[base][0],
                "tgt": MINIMAL_PAIR_SYLLABLES[base][0],
            })
    false_hits = 0
    correct_misses = 0
    total = 0
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audio_by_key:
                continue
            fp = cache_ngang.fingerprint(audio_by_key[k], sr)
            h = cache_ngang.lookup_existence(fp, "en")
            total += 1
            if h is None:
                correct_misses += 1
            else:
                false_hits += 1

    # VIVOS positive-pair recall under existence rule:
    # treat each pair (a, b) as "store fp(b), query fp(a), check hit".
    vivos_hits = 0
    vivos_total = len(vivos_fp_pairs)
    for fp_a, fp_b in vivos_fp_pairs:
        # apply existence rule directly on the two fingerprints, equivalent to
        # a singleton-store cache lookup. Avoid recomputing fingerprints.
        t = float(np.dot(fp_a.tone,    fp_b.tone))
        s = float(np.dot(fp_a.segment, fp_b.segment))
        if t >= tau_A and s >= tau_B:
            vivos_hits += 1
    vivos_recall = vivos_hits / max(vivos_total, 1)

    return {
        "tau_A": round(tau_A, 4),
        "self_lookup": [self_hits, self_total],
        "correct_miss": [correct_misses, total],
        "false_hit_rate":   round(false_hits / max(total, 1), 4),
        "vivos_pos_recall": round(vivos_recall, 4),
        "vivos_pos_hits":   [vivos_hits, vivos_total],
    }


def main() -> int:
    print("[load] MMS-TTS-vie ...")
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
                print(f"  [warn] synth fail {v}: {e}")
    print(f"[synth] {len(audio_by_key)} minimal-pair clips at sr={sr}")

    print(f"[vivos] building {N_VIVOS_POS_PAIRS} positive pairs (seed={SEED}) ...")
    pos_pairs = _build_vivos_positive_pairs(target=N_VIVOS_POS_PAIRS, seed=SEED)
    print(f"[vivos] got {len(pos_pairs)} same-text-diff-speaker pairs")

    # Precompute fingerprints for every unique VIVOS wav once
    fp_helper = TwoStreamCache()
    fp_cache: dict = {}
    def _get_fp(p: Path):
        if p in fp_cache: return fp_cache[p]
        a, asr = _load_audio(p)
        fp = fp_helper.fingerprint(a, asr)
        fp_cache[p] = fp
        return fp

    unique_wavs = {p for pair in pos_pairs for p in pair}
    print(f"[fp] computing two-stream fingerprints on {len(unique_wavs)} unique VIVOS clips ...")
    for i, p in enumerate(sorted(unique_wavs, key=lambda x: str(x))):
        _get_fp(p)
        if (i + 1) % 50 == 0:
            print(f"  fp {i+1}/{len(unique_wavs)}")

    vivos_fp_pairs = [(_get_fp(a), _get_fp(b)) for (a, b) in pos_pairs]

    print(f"\n[sweep] tau_A grid = {TAU_A_GRID}, tau_B = {TAU_B}")
    rows = []
    for tau_A in TAU_A_GRID:
        r = _eval_existence_rule(audio_by_key, sr, vivos_fp_pairs, tau_A=tau_A, tau_B=TAU_B)
        rows.append(r)
        print(f"  tau_A={tau_A:.2f}  self={r['self_lookup'][0]}/{r['self_lookup'][1]}"
              f"  corr-miss={r['correct_miss'][0]}/{r['correct_miss'][1]}"
              f"  FHR={r['false_hit_rate']:.3f}"
              f"  vivos-recall={r['vivos_pos_recall']:.3f}")

    # Recommendation: first (smallest) tau_A with correct_miss >= 24/25 AND vivos_recall >= MIN_VIVOS_RECALL.
    recommended = None
    for r in rows:
        if r["correct_miss"][0] >= MIN_CORRECT_MISS and r["vivos_pos_recall"] >= MIN_VIVOS_RECALL:
            recommended = r["tau_A"]
            break

    if recommended is None:
        # No tau_A satisfies both criteria; recommend the original 0.85.
        rec_row = rows[0]
        headline = (f"NO tau_A in {TAU_A_GRID} achieves >= {MIN_CORRECT_MISS}/25 correct-miss "
                    f"with >= {MIN_VIVOS_RECALL*100:.0f}% VIVOS recall; "
                    f"recommend retaining tau_A=0.85 ({rec_row['correct_miss'][0]}/25 correct-miss, "
                    f"VIVOS recall {rec_row['vivos_pos_recall']:.2f}).")
        recommended = 0.85
    else:
        rec_row = next(r for r in rows if r["tau_A"] == recommended)
        headline = (f"existence rule at tau_A={recommended:.2f} tau_B={TAU_B} achieves "
                    f"{rec_row['correct_miss'][0]}/25 correct-miss with "
                    f"{rec_row['self_lookup'][0]}/{rec_row['self_lookup'][1]} self-lookup retention "
                    f"and VIVOS-positive recall {rec_row['vivos_pos_recall']:.2f}.")

    out = {
        "rule":            "existence",
        "tau_B":           TAU_B,
        "seed":            SEED,
        "n_vivos_pos_pairs": len(pos_pairs),
        "sweep":           rows,
        "recommended_tau_A": recommended,
        "criteria": {
            "min_correct_miss_out_of_25": MIN_CORRECT_MISS,
            "min_vivos_pos_recall":       MIN_VIVOS_RECALL,
        },
        "headline":        headline,
    }
    out_path = PROJECT / "Results" / "tau_A_tightening.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    print(f"[headline] {headline}")

    # ---------------- Figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [r["tau_A"] for r in rows]
    cm_rate = [r["correct_miss"][0] / r["correct_miss"][1] for r in rows]
    vivos_r = [r["vivos_pos_recall"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.set_xlabel(r"$\tau_A$ (tone-stream threshold)")
    ax1.set_ylabel("Correct-miss rate (tonal variants, 25 pairs)", color="tab:red")
    ax1.plot(taus, cm_rate, marker="o", color="tab:red", linewidth=2,
             label="Correct-miss rate")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(MIN_CORRECT_MISS / 25.0, color="tab:red", linestyle=":",
                alpha=0.45, label=f"safety floor ({MIN_CORRECT_MISS}/25)")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("VIVOS positive-pair recall (200 pairs)", color="tab:blue")
    ax2.plot(taus, vivos_r, marker="s", color="tab:blue", linewidth=2,
             label="VIVOS positive recall")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0.0, 1.05)

    if recommended is not None:
        ax1.axvline(recommended, color="black", linestyle="--", alpha=0.5)
        ax1.annotate(f"recommended $\\tau_A={recommended:.2f}$",
                     xy=(recommended, 0.05), xytext=(recommended + 0.005, 0.10),
                     fontsize=9)

    fig.suptitle(r"Existence rule: $\tau_A$ tightening sweep (tau_B = 0.93)",
                 fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig_path = PROJECT / "papers" / "figures" / "fig_tau_a_tightening.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
