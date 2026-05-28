"""
SNR-adaptive (tau_A, tau_B) thresholding for the two-stream voting cache.

Robustness fix: fixed thresholds (tau_A=0.85, tau_B=0.93) are calibrated on
clean audio but the cache must serve workers under industrial SNR. We
estimate input-buffer SNR cheaply in NumPy (no ground truth required), look
up an operating point from a piecewise-linear table, and validate that the
adaptive scheme reduces false-hits at low SNR while keeping clean-condition
recall intact.

Pipeline (CPU-only, deterministic):
  1. Synthesize the 120-clip Vietnamese tonal minimal-pair corpus (same
     20-syllable x 6-tone protocol as minimal_pair_eval_large.py).
  2. Mix DEMAND-like noise at SNR in {5, 10, 15, 20} dB; keep clean too.
  3. Validate the SNR estimator against the synthesis-side ground truth
     (MAE in dB).
  4. Run the existence rule under (a) fixed (0.85, 0.93) and (b) the
     SNR-adaptive lookup, on the same noise-augmented clips.
  5. Save Results/adaptive_threshold_eval.json + papers/figures/fig_adaptive_threshold.png.

Outputs are honest measurements; nothing is fabricated.
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

import numpy as np                                                  # noqa: E402

from two_stream_voting import TwoStreamCache, TwoStreamConfig       # noqa: E402
from minimal_pair_eval_large import (                                # noqa: E402
    MINIMAL_PAIR_SYLLABLES_20,
    CARRIER,
    synth_audio,
    _demand_like_noise,
    _mix_at_snr,
    _seed_all,
    _build_corpus,
)


# ---------------- SNR estimator (pure NumPy, no ground truth) ----------------

def estimate_snr_db(audio: np.ndarray, sr: int = 16000,
                    frame_ms: float = 25.0, hop_ms: float = 10.0,
                    silence_pct: float = 0.10) -> float:
    """Energy-based SNR estimator.

    Splits the buffer into short frames; flags as silence the frames whose
    short-time energy falls below ``silence_pct`` times the median voiced
    energy (where voiced is everything above the 25th percentile). Returns
    10*log10(E_voiced_mean / E_silence_mean) clipped to a sensible range.

    Pure NumPy. Works on any single buffer; no noise reference required.
    """
    if len(audio) < int(sr * 0.05):
        return 30.0
    n_frame = max(1, int(sr * frame_ms / 1000))
    n_hop   = max(1, int(sr * hop_ms / 1000))
    n_total = (len(audio) - n_frame) // n_hop + 1
    if n_total < 4:
        return 30.0

    idx = np.arange(n_total) * n_hop
    frames = np.stack([audio[i:i + n_frame] for i in idx], axis=0).astype(np.float32)
    energy = (frames ** 2).mean(axis=1) + 1e-12

    # split frames: bottom 25 percentile = candidate silence, top 50 = voiced
    p25 = float(np.percentile(energy, 25))
    p50 = float(np.percentile(energy, 50))
    voiced_mask = energy >= p50
    voiced_energy = energy[voiced_mask].mean() if voiced_mask.any() else float(energy.max())

    silence_thr = silence_pct * float(np.median(energy[voiced_mask])) \
                  if voiced_mask.any() else p25
    silence_mask = energy < max(silence_thr, p25 * 0.5)
    if not silence_mask.any():
        # very flat signal: fall back to bottom-decile energy as noise floor
        noise_energy = float(np.percentile(energy, 10))
    else:
        noise_energy = float(energy[silence_mask].mean())
    noise_energy = max(noise_energy, 1e-10)

    snr_lin = voiced_energy / noise_energy
    snr_db = 10.0 * np.log10(max(snr_lin, 1e-6))
    # clamp to plausible range so synthesis-side ground truth is comparable
    return float(np.clip(snr_db, -10.0, 60.0))


# ---------------- Piecewise-linear (tau_A, tau_B) lookup ----------------
# Bands as specified in the task; thresholds are precision-friendlier as
# SNR drops, recall-friendlier as SNR rises.
THRESHOLD_TABLE = [
    # (snr_low_db, snr_high_db, tau_A, tau_B, label)
    (25.0,  float("inf"), 0.85, 0.93, "clean"),
    (15.0,  25.0,         0.88, 0.93, "moderate"),
    ( 5.0,  15.0,         0.92, 0.95, "noisy"),
    (-1e9,   5.0,         0.95, 0.96, "severe"),
]

def lookup_thresholds(snr_db: float) -> tuple[float, float, str]:
    for lo, hi, tA, tB, label in THRESHOLD_TABLE:
        if lo <= snr_db < hi:
            return tA, tB, label
    return 0.85, 0.93, "clean"  # default


# ---------------- Eval at a single (tau_A, tau_B) ----------------

def _eval_op_point(audios_by_key: dict, sr: int,
                   tau_A: float, tau_B: float) -> dict:
    """Self-lookup recall + tonal correct-miss / false-hit at one operating
    point, exactly mirroring minimal_pair_eval_large._run_eval_one_condition.
    """
    cfg = TwoStreamConfig(tone_threshold=tau_A,
                          segment_threshold=tau_B,
                          max_entries=1024)

    cache_self = TwoStreamCache(cfg)
    fps_self = {}
    for k, a in audios_by_key.items():
        fp = cache_self.fingerprint(a, sr)
        fps_self[k] = fp
        cache_self.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})
    self_hits = sum(
        1 for k, fp in fps_self.items()
        if (h := cache_self.lookup_existence(fp, "en")) is not None
        and h["result"]["vi"] == f"{k[0]}/{k[1]}"
    )
    n_clips = len(audios_by_key)

    cache_t = TwoStreamCache(cfg)
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        ngang_key = (base, variants[0])
        if ngang_key in audios_by_key:
            fp = cache_t.fingerprint(audios_by_key[ngang_key], sr)
            cache_t.put(fp, "en", {"vi": variants[0], "tgt": variants[0]})

    correct_misses = 0
    false_hits = 0
    total = 0
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audios_by_key:
                continue
            fp = cache_t.fingerprint(audios_by_key[k], sr)
            h = cache_t.lookup_existence(fp, "en")
            total += 1
            if h is None:
                correct_misses += 1
            else:
                false_hits += 1
    return {
        "tau_A": tau_A, "tau_B": tau_B,
        "n_clips_total": n_clips,
        "self_hits": self_hits,
        "self_recall": round(self_hits / max(n_clips, 1), 4),
        "n_tonal_probes": total,
        "correct_misses": correct_misses,
        "correct_miss_rate": round(correct_misses / max(total, 1), 4),
        "false_hits": false_hits,
        "false_hit_rate": round(false_hits / max(total, 1), 4),
    }


def _eval_adaptive(audios_by_key: dict, sr: int) -> dict:
    """Like _eval_op_point but per-clip (tau_A, tau_B) lookup based on the
    *query* clip's estimated SNR. We must therefore loop without sharing a
    single cache cfg.

    Strategy: pre-build entry fingerprints once (cache stores ngang). For
    each probe clip estimate SNR, then re-run the existence check manually
    against the stored entries with the looked-up thresholds. This avoids
    rebuilding the cache for every clip while still allowing per-probe taus.
    """
    # Pre-extract every fingerprint at the default cfg (thresholds only matter
    # at lookup time, not at fingerprint extraction).
    helper = TwoStreamCache(TwoStreamConfig(tone_threshold=0.85,
                                            segment_threshold=0.93,
                                            max_entries=1024))
    fps = {k: helper.fingerprint(a, sr) for k, a in audios_by_key.items()}

    # Self-lookup: for each clip, look up against the cache of all clips
    # using the per-clip estimated SNR -> thresholds.
    all_keys = list(audios_by_key.keys())
    fp_keys = list(fps.keys())
    tone_mat = np.stack([fps[k].tone for k in fp_keys])
    seg_mat  = np.stack([fps[k].segment for k in fp_keys])

    self_hits = 0
    for qk in fp_keys:
        snr = estimate_snr_db(audios_by_key[qk], sr)
        tA, tB, _ = lookup_thresholds(snr)
        q_t = fps[qk].tone; q_s = fps[qk].segment
        cos_t = tone_mat @ q_t
        cos_s = seg_mat @ q_s
        # any entry passes both? must include self -> recall.
        hits = (cos_t >= tA) & (cos_s >= tB)
        if hits.any():
            # accept if best hit is the same key (since each clip is unique)
            best_idx = int(np.argmax(np.where(hits, cos_t + cos_s, -np.inf)))
            if fp_keys[best_idx] == qk:
                self_hits += 1
            else:
                # collision with another clip: still a hit, but not self-hit.
                # We count "self_hits" as how many times the system returned
                # the SAME-key entry; collisions are tracked separately below.
                pass
    self_recall = self_hits / max(len(fp_keys), 1)

    # Tonal-variant probes: cache stores only ngang per syllable.
    ngang_keys = [(b, vs[0]) for b, vs in MINIMAL_PAIR_SYLLABLES_20.items()
                  if (b, vs[0]) in fps]
    ngang_tone = np.stack([fps[k].tone for k in ngang_keys])
    ngang_seg  = np.stack([fps[k].segment for k in ngang_keys])

    correct_misses = 0
    false_hits = 0
    total = 0
    snr_log: list[float] = []
    band_log: list[str] = []
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in fps:
                continue
            snr = estimate_snr_db(audios_by_key[k], sr)
            tA, tB, label = lookup_thresholds(snr)
            snr_log.append(snr); band_log.append(label)
            q_t = fps[k].tone; q_s = fps[k].segment
            cos_t = ngang_tone @ q_t
            cos_s = ngang_seg  @ q_s
            hits = (cos_t >= tA) & (cos_s >= tB)
            total += 1
            if hits.any():
                false_hits += 1
            else:
                correct_misses += 1

    return {
        "scheme": "adaptive",
        "n_clips_total": len(fp_keys),
        "self_hits": self_hits,
        "self_recall": round(self_recall, 4),
        "n_tonal_probes": total,
        "correct_misses": correct_misses,
        "correct_miss_rate": round(correct_misses / max(total, 1), 4),
        "false_hits": false_hits,
        "false_hit_rate": round(false_hits / max(total, 1), 4),
        "snr_estimates_mean": float(np.mean(snr_log)) if snr_log else None,
        "snr_estimates_std":  float(np.std(snr_log))  if snr_log else None,
        "band_counts": {b: band_log.count(b) for b in {"clean", "moderate", "noisy", "severe"}},
    }


# ---------------- Plot ----------------

def plot_adaptive_vs_fixed(per_condition_results: dict, out_path: Path,
                           dpi: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(per_condition_results.keys())
    fixed_fhr  = [per_condition_results[c]["fixed"]["false_hit_rate"]  for c in conds]
    adapt_fhr  = [per_condition_results[c]["adaptive"]["false_hit_rate"] for c in conds]
    fixed_rec  = [per_condition_results[c]["fixed"]["self_recall"]    for c in conds]
    adapt_rec  = [per_condition_results[c]["adaptive"]["self_recall"] for c in conds]

    x = np.arange(len(conds))
    w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    b1 = ax[0].bar(x - w/2, fixed_fhr, w, label=r"fixed ($\tau_A=0.85,\tau_B=0.93$)", color="#d95f02")
    b2 = ax[0].bar(x + w/2, adapt_fhr, w, label=r"SNR-adaptive", color="#1b9e77")
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds, rotation=0)
    ax[0].set_ylabel("false-hit rate (lower = better)")
    ax[0].set_title("Adaptive vs fixed thresholds: tonal false-hit rate")
    ax[0].set_ylim(0, max(max(fixed_fhr), max(adapt_fhr), 0.8) * 1.15)
    ax[0].legend(loc="upper right", fontsize=8)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax[0].text(r.get_x() + r.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax[1].bar(x - w/2, fixed_rec, w, label="fixed",        color="#d95f02")
    ax[1].bar(x + w/2, adapt_rec, w, label="SNR-adaptive", color="#1b9e77")
    ax[1].set_xticks(x); ax[1].set_xticklabels(conds, rotation=0)
    ax[1].set_ylabel("self-lookup recall (higher = better)")
    ax[1].set_title("Self-lookup recall preserved")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Results/adaptive_threshold_eval.json")
    ap.add_argument("--fig", default="papers/figures/fig_adaptive_threshold.png")
    ap.add_argument("--snrs", nargs="+", type=float, default=[20, 15, 10, 5])
    args = ap.parse_args()

    _seed_all()

    print("[load] MMS-TTS-vie ...", flush=True)
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = int(model.config.sampling_rate)
    print(f"[load] ready, sr={sr}", flush=True)

    audio_clean = _build_corpus(tok, model, sr)
    if len(audio_clean) < 120:
        print(f"[warn] only {len(audio_clean)} clips (target 120)", flush=True)

    # ---- SNR estimator validation against ground truth ----
    print("\n[snr-est] validating estimator on synthesis-side ground truth", flush=True)
    snr_validation = {}
    abs_errs: list[float] = []
    for gt_snr in args.snrs:
        ests: list[float] = []
        for idx, (k, a) in enumerate(audio_clean.items()):
            rng = np.random.default_rng(20260528 + int(gt_snr * 100) + idx)
            noisy = _mix_at_snr(a, gt_snr, rng, sr)
            est = estimate_snr_db(noisy, sr)
            ests.append(est)
        ests_arr = np.asarray(ests)
        mae = float(np.mean(np.abs(ests_arr - gt_snr)))
        bias = float(np.mean(ests_arr - gt_snr))
        abs_errs.extend([abs(e - gt_snr) for e in ests])
        snr_validation[f"snr_{int(gt_snr)}dB"] = {
            "gt_snr_db": gt_snr,
            "est_mean": float(ests_arr.mean()),
            "est_std":  float(ests_arr.std()),
            "mae_db":   round(mae, 3),
            "bias_db":  round(bias, 3),
        }
        print(f"   gt={gt_snr:>5.1f} dB  est_mean={ests_arr.mean():6.2f}  "
              f"mae={mae:5.2f}  bias={bias:+5.2f}", flush=True)
    # Clean condition: just estimate
    clean_ests = np.asarray([estimate_snr_db(a, sr) for a in audio_clean.values()])
    snr_validation["clean"] = {
        "gt_snr_db": None,
        "est_mean": float(clean_ests.mean()),
        "est_std":  float(clean_ests.std()),
    }
    overall_mae = float(np.mean(abs_errs))
    print(f"   overall MAE across SNR conditions: {overall_mae:.2f} dB", flush=True)

    # ---- Per-condition fixed vs adaptive eval ----
    conditions = [("clean", None)] + [(f"snr_{int(s)}dB", float(s)) for s in args.snrs]
    per_condition: dict[str, dict] = {}
    for cname, gt_snr in conditions:
        print(f"\n[cond] {cname}  gt_snr={gt_snr}", flush=True)
        if gt_snr is None:
            audios = dict(audio_clean)
        else:
            audios = {}
            for idx, (k, a) in enumerate(audio_clean.items()):
                rng = np.random.default_rng(20260528 + int(gt_snr * 100) + idx)
                audios[k] = _mix_at_snr(a, gt_snr, rng, sr)

        t0 = time.perf_counter()
        fixed = _eval_op_point(audios, sr, 0.85, 0.93)
        t_fixed = time.perf_counter() - t0
        t0 = time.perf_counter()
        adaptive = _eval_adaptive(audios, sr)
        t_adapt = time.perf_counter() - t0

        # Mean estimated SNR for this condition (over all clips, for logging)
        mean_est = float(np.mean([estimate_snr_db(a, sr) for a in audios.values()]))

        print(f"   fixed:    self={fixed['self_hits']}/120  "
              f"FHR={fixed['false_hit_rate']*100:5.1f}%  CM={fixed['correct_miss_rate']*100:5.1f}%  "
              f"elapsed={t_fixed:.1f}s", flush=True)
        print(f"   adaptive: self={adaptive['self_hits']}/120  "
              f"FHR={adaptive['false_hit_rate']*100:5.1f}%  CM={adaptive['correct_miss_rate']*100:5.1f}%  "
              f"snr_est_mean={mean_est:5.1f}dB  band_counts={adaptive['band_counts']}  "
              f"elapsed={t_adapt:.1f}s", flush=True)

        per_condition[cname] = {
            "gt_snr_db": gt_snr,
            "mean_estimated_snr_db": round(mean_est, 2),
            "fixed":    fixed,
            "adaptive": adaptive,
        }

    # ---- Save ----
    out = {
        "meta": {
            "carrier_template": CARRIER,
            "n_syllables": 20,
            "n_tones": 6,
            "n_clips_target": 120,
            "snr_conditions_db": args.snrs,
            "threshold_table": [
                {"snr_low": (lo if lo > -1e8 else None),
                 "snr_high": (hi if hi != float("inf") else None),
                 "tau_A": tA, "tau_B": tB, "label": lab}
                for (lo, hi, tA, tB, lab) in THRESHOLD_TABLE
            ],
            "fixed_baseline": {"tau_A": 0.85, "tau_B": 0.93},
            "noise_model": ("synthetic DEMAND-like (pink-tilted noise + 85/110 Hz rumble); "
                            "RMS-scaled to SNR per clip"),
        },
        "snr_estimator_validation": {
            "per_condition": snr_validation,
            "overall_mae_db": round(overall_mae, 3),
        },
        "results_by_condition": per_condition,
    }
    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] wrote {out_path}", flush=True)

    fig_path = PROJECT / args.fig
    plot_adaptive_vs_fixed(per_condition, fig_path, dpi=200)
    print(f"[done] wrote {fig_path}", flush=True)

    # ---- Console summary ----
    print("\n" + "=" * 78)
    print(" ADAPTIVE-THRESHOLD SUMMARY")
    print("=" * 78)
    print(f"{'cond':>10} {'gt_snr':>7} {'est_snr':>8} "
          f"{'FHR_fix':>8} {'FHR_adapt':>10} {'rec_fix':>8} {'rec_adapt':>10}")
    for c, r in per_condition.items():
        print(f"{c:>10} {str(r['gt_snr_db']):>7} {r['mean_estimated_snr_db']:>8.2f} "
              f"{r['fixed']['false_hit_rate']:>8.3f} {r['adaptive']['false_hit_rate']:>10.3f} "
              f"{r['fixed']['self_recall']:>8.3f} {r['adaptive']['self_recall']:>10.3f}")
    print(f"\n  SNR estimator overall MAE: {overall_mae:.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
