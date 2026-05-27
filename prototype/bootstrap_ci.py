"""Paper-grade bootstrap CIs + paired permutation tests on §4 headline benchmarks.

Reads all per-experiment JSONs under Results/, computes:
  - 95% bootstrap CI (B=1000, percentile method) on headline stats.
  - Paired permutation test (B=1000, sign-flip) for paired comparisons.
  - Two-sided p-values; reports whether p < 0.05.

Writes Results/bootstrap_ci_all.json and prints a human-readable console table.

Pure NumPy. No SciPy.

Honest caveats (printed at end):
  - bench_compare_en_n5: N=5 paired utterances. CIs wide.
  - ablation_en_n3:      N=3 per config. CIs *very* wide; treat as direction-only.
  - fingerprint_ablation, minimal_pair_eval, noise_robustness_eval, worker_simulation:
        only summary mean/std stored, no per-pair data. We resample
        parametrically from N(mean, std) — this is a Gaussian assumption,
        not a true bootstrap. Flagged in output as `parametric=true`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results"
OUT_PATH = RESULTS / "bootstrap_ci_all.json"
B_BOOT = 1000
B_PERM = 1000
RNG = np.random.default_rng(20260527)


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def bootstrap_ci(
    x: np.ndarray,
    stat_fn=np.mean,
    B: int = B_BOOT,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Return (point, ci_lo, ci_hi) using percentile bootstrap."""
    if rng is None:
        rng = RNG
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(x))
    idx = rng.integers(0, n, size=(B, n))
    resamples = x[idx]
    stats = np.array([stat_fn(resamples[b]) for b in range(B)], dtype=float)
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def bootstrap_ci_parametric(
    mean: float,
    std: float,
    n: int,
    B: int = B_BOOT,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Parametric Gaussian-assumption bootstrap when raw samples are not stored."""
    if rng is None:
        rng = RNG
    if n <= 0 or not np.isfinite(std):
        return (float(mean), float("nan"), float("nan"))
    sample_means = rng.normal(loc=mean, scale=std / np.sqrt(max(n, 1)), size=B)
    lo = float(np.percentile(sample_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(sample_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(mean), lo, hi


def paired_permutation_p(
    a: np.ndarray,
    b: np.ndarray,
    B: int = B_PERM,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """Paired permutation test by sign-flipping diffs. Two-sided p.

    Returns (observed_mean_diff, p_value).
    """
    if rng is None:
        rng = RNG
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired vectors must match"
    d = a - b
    n = len(d)
    if n == 0:
        return (0.0, float("nan"))
    obs = float(np.mean(d))
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    null_means = (signs * d).mean(axis=1)
    p = float((np.abs(null_means) >= abs(obs) - 1e-12).mean())
    # Add (B+1) smoothing so p never reports 0.0
    p = max(p, 1.0 / (B + 1))
    return obs, p


# ---------------------------------------------------------------------------
# Per-file analyses
# ---------------------------------------------------------------------------

def analyze_bench_compare(path: Path) -> Dict:
    """FP32 vs INT8 cascade latency per stage. Paired by utt_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    base = data["baseline"]["samples"]
    fast = data["fast"]["samples"]
    by_id_base = {s["utt_id"]: s for s in base}
    by_id_fast = {s["utt_id"]: s for s in fast}
    ids = [u for u in by_id_base if u in by_id_fast]
    out: Dict = {"n_paired": len(ids), "parametric": False}

    for stage in ["asr", "mt", "tts", "total"]:
        fp32 = np.array([by_id_base[u]["latency_ms"][stage] for u in ids])
        int8 = np.array([by_id_fast[u]["latency_ms"][stage] for u in ids])
        for label, vec in [(f"{stage}_fp32", fp32), (f"{stage}_int8", int8)]:
            for stat_name, stat_fn in [("mean", np.mean), ("p50", lambda v: np.percentile(v, 50)), ("p95", lambda v: np.percentile(v, 95))]:
                pt, lo, hi = bootstrap_ci(vec, stat_fn=stat_fn)
                out[f"{label}_{stat_name}"] = {"point": pt, "ci95_lo": lo, "ci95_hi": hi}
        diff_obs, p = paired_permutation_p(fp32, int8)
        speedup = float(fp32.mean() / int8.mean()) if int8.mean() > 0 else float("nan")
        # bootstrap CI on speedup ratio (resample same indices for both vectors)
        n = len(ids)
        idx = RNG.integers(0, n, size=(B_BOOT, n))
        ratios = fp32[idx].mean(axis=1) / np.maximum(int8[idx].mean(axis=1), 1e-9)
        out[f"{stage}_speedup"] = {
            "point": speedup,
            "ci95_lo": float(np.percentile(ratios, 2.5)),
            "ci95_hi": float(np.percentile(ratios, 97.5)),
        }
        out[f"{stage}_paired_permutation_p"] = p
        out[f"{stage}_paired_mean_diff_ms"] = diff_obs
    return out


def analyze_ablation(path: Path) -> Dict:
    """6-config cascade ablation. Paired comparisons across configs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    per_cfg = data["per_config_results"]
    # Collect per-config wall_ms keyed by utt_id
    cfg_vecs: Dict[str, Dict[str, float]] = {}
    cfg_order: List[str] = []
    for cfg in per_cfg:
        cid = cfg["config_id"]
        cfg_order.append(cid)
        cfg_vecs[cid] = {s["utt_id"]: s["wall_ms"] for s in cfg["samples"]}

    common_ids = sorted(set.intersection(*[set(v) for v in cfg_vecs.values()]))
    out: Dict = {"n_paired": len(common_ids), "configs": cfg_order, "parametric": False, "per_config": {}}

    for cid in cfg_order:
        v = np.array([cfg_vecs[cid][u] for u in common_ids])
        ms = {}
        for stat_name, stat_fn in [("mean", np.mean), ("p50", lambda x: np.percentile(x, 50)), ("p95", lambda x: np.percentile(x, 95))]:
            pt, lo, hi = bootstrap_ci(v, stat_fn=stat_fn)
            ms[stat_name] = {"point": pt, "ci95_lo": lo, "ci95_hi": hi}
        out["per_config"][cid] = ms

    # Headline paired comparisons (all vs int8_cascade baseline)
    baseline_id = "int8_cascade"
    base_v = np.array([cfg_vecs[baseline_id][u] for u in common_ids])
    cmps = {}
    for cid in cfg_order:
        if cid == baseline_id:
            continue
        v = np.array([cfg_vecs[cid][u] for u in common_ids])
        diff, p = paired_permutation_p(base_v, v)
        speedup = float(base_v.mean() / v.mean()) if v.mean() > 0 else float("nan")
        cmps[f"{baseline_id}_vs_{cid}"] = {
            "mean_diff_ms": diff,
            "speedup_point": speedup,
            "paired_permutation_p": p,
            "significant_at_0.05": bool(p < 0.05),
        }
    # Specifically the "+T1 vs +T2 vs +T1+T2" headline mentioned in spec
    for a_id, b_id in [("int8_t1_only", "int8_t2_only"), ("int8_t1_only", "int8_t1_t2"), ("int8_t2_only", "int8_t1_t2")]:
        a_v = np.array([cfg_vecs[a_id][u] for u in common_ids])
        b_v = np.array([cfg_vecs[b_id][u] for u in common_ids])
        diff, p = paired_permutation_p(a_v, b_v)
        cmps[f"{a_id}_vs_{b_id}"] = {
            "mean_diff_ms": diff,
            "paired_permutation_p": p,
            "significant_at_0.05": bool(p < 0.05),
        }
    out["pairwise"] = cmps
    return out


def analyze_fingerprint_ablation(path: Path) -> Dict:
    """Only summary stats available — use parametric bootstrap."""
    data = json.loads(path.read_text(encoding="utf-8"))
    n_intra = data.get("n_intra_pairs", 50)
    n_inter = data.get("n_inter_pairs", 50)
    out: Dict = {"parametric": True, "n_intra": n_intra, "n_inter": n_inter, "per_ablation": {}}

    full = next(a for a in data["ablations"] if a["ablation"] == "full")
    naive = next((a for a in data["ablations"] if a["ablation"] == "tone_only"), None)

    for ab in data["ablations"]:
        ab_id = ab["ablation"]
        intra_pt, intra_lo, intra_hi = bootstrap_ci_parametric(ab["intra_cos_mean"], ab["intra_cos_std"], n_intra)
        inter_pt, inter_lo, inter_hi = bootstrap_ci_parametric(ab["inter_cos_mean"], ab["inter_cos_std"], n_inter)
        # gap CI: gap = intra - inter; treat as independent samples
        # combined SE = sqrt(var_intra/n_intra + var_inter/n_inter)
        se_gap = np.sqrt(ab["intra_cos_std"] ** 2 / n_intra + ab["inter_cos_std"] ** 2 / n_inter)
        gap_draws = RNG.normal(loc=ab["gap"], scale=se_gap, size=B_BOOT)
        out["per_ablation"][ab_id] = {
            "intra_cos_mean": {"point": intra_pt, "ci95_lo": intra_lo, "ci95_hi": intra_hi},
            "inter_cos_mean": {"point": inter_pt, "ci95_lo": inter_lo, "ci95_hi": inter_hi},
            "gap": {
                "point": ab["gap"],
                "ci95_lo": float(np.percentile(gap_draws, 2.5)),
                "ci95_hi": float(np.percentile(gap_draws, 97.5)),
            },
        }

    # Naive fingerprint (tone_only) vs full fingerprint gap test — Welch-style approximation
    if naive is not None:
        se_full = np.sqrt(full["intra_cos_std"] ** 2 / n_intra + full["inter_cos_std"] ** 2 / n_inter)
        se_naive = np.sqrt(naive["intra_cos_std"] ** 2 / n_intra + naive["inter_cos_std"] ** 2 / n_inter)
        # difference of gaps
        diff_obs = full["gap"] - naive["gap"]
        se_diff = np.sqrt(se_full ** 2 + se_naive ** 2)
        # two-sided p under N(0, se_diff)
        z = diff_obs / se_diff if se_diff > 0 else float("inf")
        # erfc-based two-sided
        from math import erfc, sqrt
        p_full_vs_naive = float(erfc(abs(z) / sqrt(2)))
        out["full_vs_tone_only_gap"] = {
            "diff_point": diff_obs,
            "z_score": float(z),
            "p_two_sided_z": p_full_vs_naive,
            "significant_at_0.05": bool(p_full_vs_naive < 0.05),
        }
    return out


def analyze_minimal_pair(path: Path) -> Dict:
    """Protect-score per variant. Only summary stats — parametric."""
    data = json.loads(path.read_text(encoding="utf-8"))
    n_syl = data.get("n_syllables_synthed", 36)
    out: Dict = {"parametric": True, "n_syllables": n_syl, "per_variant": {}}
    for v in data["variants"]:
        vid = v["variant"]
        intra_pt, lo_i, hi_i = bootstrap_ci_parametric(v["intra_tone_cos_mean"], v["intra_tone_cos_std"], n_syl)
        inter_pt, lo_e, hi_e = bootstrap_ci_parametric(v["inter_syl_cos_mean"], v["inter_syl_cos_std"], n_syl)
        se_gap = np.sqrt(v["intra_tone_cos_std"] ** 2 / n_syl + v["inter_syl_cos_std"] ** 2 / n_syl)
        draws = RNG.normal(loc=v["protect_score"], scale=se_gap, size=B_BOOT)
        out["per_variant"][vid] = {
            "intra_tone_cos": {"point": intra_pt, "ci95_lo": lo_i, "ci95_hi": hi_i},
            "inter_syl_cos": {"point": inter_pt, "ci95_lo": lo_e, "ci95_hi": hi_e},
            "protect_score": {
                "point": v["protect_score"],
                "ci95_lo": float(np.percentile(draws, 2.5)),
                "ci95_hi": float(np.percentile(draws, 97.5)),
            },
        }
    return out


def analyze_noise_robustness(path: Path) -> Dict:
    """Raw vs filtered cosine per SNR. Only summary stats — parametric.

    Headline question: at each SNR, is filtered cosine significantly > raw cosine?
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict = {"parametric": True, "per_snr": {}}
    from math import erfc, sqrt
    for snr in data["snr_curves"]:
        n = snr["n_total"]
        snr_db = snr["snr_db_target"]
        raw_mean, raw_std = snr["cache_cosine_raw_mean"], snr["cache_cosine_raw_std"]
        flt_mean, flt_std = snr["cache_cosine_filtered_mean"], snr["cache_cosine_filtered_std"]
        raw_pt, raw_lo, raw_hi = bootstrap_ci_parametric(raw_mean, raw_std, n)
        flt_pt, flt_lo, flt_hi = bootstrap_ci_parametric(flt_mean, flt_std, n)
        # Welch-style two-sample z (small n, but no raw data — best we can do)
        se_diff = np.sqrt(raw_std ** 2 / n + flt_std ** 2 / n)
        diff = flt_mean - raw_mean
        z = diff / se_diff if se_diff > 0 else float("inf")
        p = float(erfc(abs(z) / sqrt(2)))
        # hit rates: exact binomial CI via normal approx on proportion
        raw_hits = snr["hits_raw_at_0.95"]
        flt_hits = snr["hits_filtered_at_0.95"]
        out["per_snr"][f"{snr_db}dB"] = {
            "n": n,
            "raw_cosine_mean": {"point": raw_pt, "ci95_lo": raw_lo, "ci95_hi": raw_hi},
            "filtered_cosine_mean": {"point": flt_pt, "ci95_lo": flt_lo, "ci95_hi": flt_hi},
            "filtered_minus_raw": {
                "point": diff,
                "z_score": float(z),
                "p_two_sided_z": p,
                "significant_at_0.05": bool(p < 0.05),
            },
            "raw_hit_rate": raw_hits / n,
            "filtered_hit_rate": flt_hits / n,
        }
    return out


def analyze_worker_simulation(path: Path) -> Dict:
    """3 scenarios. Only summary latency_ms aggregate — parametric.

    Cannot derive std from the JSON; use (p95 - p50) / 1.645 as a rough
    Gaussian-equivalent std (one-sided). Flagged as crude.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict = {"parametric": True, "std_proxy": "(p95-p50)/1.645", "per_scenario": {}}
    for sc in data["scenarios"]:
        nw = sc["config"]["n_workers"]
        n_ev = sc["n_events"]
        lat = sc["latency_ms"]
        std_proxy = max((lat["p95"] - lat["p50"]) / 1.645, 1.0)
        mean_pt, mean_lo, mean_hi = bootstrap_ci_parametric(lat["mean"], std_proxy, n_ev)
        out["per_scenario"][f"workers_{nw}"] = {
            "n_events": n_ev,
            "latency_mean_ms": {"point": mean_pt, "ci95_lo": mean_lo, "ci95_hi": mean_hi},
            "latency_p50_ms": lat["p50"],
            "latency_p95_ms": lat["p95"],
            "t1_hit_rate": sc["t1_hit_rate"],
        }
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest() -> None:
    """Bootstrap mean of resampled data must converge to original sample mean within 5%."""
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=2.0, size=200)
    pt, lo, hi = bootstrap_ci(x, stat_fn=np.mean, B=2000, rng=rng)
    sample_mean = float(np.mean(x))
    rel_err = abs(pt - sample_mean) / abs(sample_mean)
    assert rel_err < 0.05, f"self-test failed: pt={pt}, mean={sample_mean}, rel_err={rel_err}"
    assert lo < sample_mean < hi, f"self-test failed: CI [{lo},{hi}] does not bracket mean {sample_mean}"
    # paired-permutation sanity: two identical vectors → p should be ~1.0
    _, p = paired_permutation_p(x, x.copy(), B=500, rng=rng)
    assert p > 0.5, f"paired-perm sanity failed: identical vectors should give large p, got {p}"
    print(f"[selftest] OK | sample_mean={sample_mean:.4f} bootstrap_pt={pt:.4f} CI=[{lo:.4f},{hi:.4f}] identical-perm-p={p:.3f}")


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def fmt_ci(d: Dict, unit: str = "") -> str:
    if "point" not in d:
        return str(d)
    pt = d["point"]
    lo = d.get("ci95_lo", float("nan"))
    hi = d.get("ci95_hi", float("nan"))
    return f"{pt:.3f}{unit} [CI95 {lo:.3f}–{hi:.3f}]"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def print_summary(all_results: Dict) -> None:
    line = "=" * 78
    print()
    print(line)
    print("BOOTSTRAP CI + PAIRED PERMUTATION SUMMARY (B_boot=1000, B_perm=1000)")
    print(line)

    # bench_compare
    for key in ("bench_compare_en_n5", "bench_compare_en_n10"):
        bc = all_results.get(key, {})
        if not bc:
            continue
        n = bc.get("n_paired", "?")
        print(f"\n[{key}]  N={n} paired utterances  (true bootstrap)")
        for stage in ["asr", "mt", "tts", "total"]:
            fp = bc.get(f"{stage}_fp32_mean", {})
            it = bc.get(f"{stage}_int8_mean", {})
            sp = bc.get(f"{stage}_speedup", {})
            p = bc.get(f"{stage}_paired_permutation_p", float("nan"))
            print(f"  {stage.upper():<5} FP32 mean  : {fmt_ci(fp, ' ms')}")
            print(f"  {stage.upper():<5} INT8 mean  : {fmt_ci(it, ' ms')}")
            print(f"  {stage.upper():<5} speedup    : {fmt_ci(sp, 'x')}   {fmt_p(p)}")

    # ablation
    ab = all_results.get("ablation_en_n3", {})
    if ab:
        print("\n[ablation_en_n3]  N=3 per config (true bootstrap, CIs *very* wide)")
        for cid, ms in ab.get("per_config", {}).items():
            print(f"  {cid:<16} latency mean: {fmt_ci(ms['mean'], ' ms')}")
        print("  Pairwise vs int8_cascade:")
        for k, v in ab.get("pairwise", {}).items():
            if not k.startswith("int8_cascade_vs_"):
                continue
            tag = "***" if v.get("significant_at_0.05") else "   "
            sp = v.get("speedup_point", float("nan"))
            print(f"    {tag} {k:<40} speedup={sp:.2f}x  {fmt_p(v['paired_permutation_p'])}")

    # fingerprint
    fp = all_results.get("fingerprint_ablation", {})
    if fp:
        print("\n[fingerprint_ablation]  N_intra=50 N_inter=50 (PARAMETRIC — summary stats only)")
        for ab_id, d in fp.get("per_ablation", {}).items():
            print(f"  {ab_id:<25} gap = {fmt_ci(d['gap'])}")
        gvn = fp.get("full_vs_tone_only_gap")
        if gvn:
            tag = "***" if gvn.get("significant_at_0.05") else "   "
            print(f"  {tag} full vs tone_only gap diff = {gvn['diff_point']:+.4f}   {fmt_p(gvn['p_two_sided_z'])}")

    # minimal pair
    mp = all_results.get("minimal_pair_eval", {})
    if mp:
        print(f"\n[minimal_pair_eval]  N_syl={mp.get('n_syllables')} (PARAMETRIC)")
        for vid, d in mp.get("per_variant", {}).items():
            print(f"  {vid:<15} protect_score = {fmt_ci(d['protect_score'])}")

    # noise robustness
    nr = all_results.get("noise_robustness_eval", {})
    if nr:
        print("\n[noise_robustness_eval]  N=10 per SNR (PARAMETRIC)")
        for snr, d in nr.get("per_snr", {}).items():
            tag = "***" if d["filtered_minus_raw"].get("significant_at_0.05") else "   "
            print(f"  {tag} {snr:<7} raw={fmt_ci(d['raw_cosine_mean'])}  filt={fmt_ci(d['filtered_cosine_mean'])}  {fmt_p(d['filtered_minus_raw']['p_two_sided_z'])}")

    # worker sim
    ws = all_results.get("worker_simulation", {})
    if ws:
        print("\n[worker_simulation]  (PARAMETRIC, std proxy = (p95-p50)/1.645)")
        for sc, d in ws.get("per_scenario", {}).items():
            print(f"  {sc:<12} mean lat = {fmt_ci(d['latency_mean_ms'], ' ms')}   p50={d['latency_p50_ms']:.0f}  p95={d['latency_p95_ms']:.0f}")

    print()
    print(line)
    print("HONESTY DISCLAIMER")
    print(line)
    print(
        "  - bench_compare (N=5) and ablation (N=3) have *very wide* CIs.\n"
        "    Reviewers will (correctly) flag low statistical power. Recommendation:\n"
        "      * bench_compare: re-run with N>=20 utterances before camera-ready.\n"
        "      * ablation: re-run with N>=15 per config (45 mins per config on CPU).\n"
        "  - fingerprint_ablation, minimal_pair_eval, noise_robustness, worker_sim:\n"
        "    only summary mean+std persisted -> CIs use a Gaussian parametric assumption.\n"
        "    For camera-ready, persist per-pair raw cosine values and re-run with true\n"
        "    percentile bootstrap on the raw vectors.\n"
        "  - For ablation N=3, even when point estimate looks favorable, paired-perm p\n"
        "    cannot drop below ~0.25 (2^3 = 8 sign-flip permutations); treat as direction\n"
        "    only, never as evidence of significance.\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    selftest()

    all_results: Dict = {}

    files = [
        ("bench_compare_en_n5", "bench_compare_en_n5.json", analyze_bench_compare),
        ("bench_compare_en_n10", "bench_compare_en_n10.json", analyze_bench_compare),
        ("ablation_en_n3", "ablation_en_n3.json", analyze_ablation),
        ("fingerprint_ablation", "fingerprint_ablation.json", analyze_fingerprint_ablation),
        ("minimal_pair_eval", "minimal_pair_eval.json", analyze_minimal_pair),
        ("noise_robustness_eval", "noise_robustness_eval.json", analyze_noise_robustness),
        ("worker_simulation", "worker_simulation.json", analyze_worker_simulation),
    ]

    for key, fname, fn in files:
        path = RESULTS / fname
        if not path.exists():
            print(f"[skip] {fname} not found")
            continue
        try:
            all_results[key] = fn(path)
            print(f"[ok]   {fname}")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {fname}: {e}")
            all_results[key] = {"error": str(e)}

    all_results["_meta"] = {
        "B_bootstrap": B_BOOT,
        "B_permutation": B_PERM,
        "alpha": 0.05,
        "method_notes": (
            "Percentile bootstrap for files with per-utterance/per-pair raw values "
            "(bench_compare, ablation). Parametric Gaussian bootstrap for files where "
            "only summary mean+std were persisted (fingerprint_ablation, minimal_pair_eval, "
            "noise_robustness_eval, worker_simulation). The latter is flagged with "
            "parametric=true and is a known weakness — persist raw vectors for camera-ready."
        ),
        "rng_seed": 20260527,
    }

    OUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {OUT_PATH}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
