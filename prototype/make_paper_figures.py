"""Paper-quality figure generator for VoiceAI_SaiGonHub §4 results.

Reads JSON files from Results/ and writes PNG figures to papers/figures/.
Pure matplotlib — no seaborn. Run standalone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results"
OUT = ROOT / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

PALETTE = {
    "asr": "#1f77b4",
    "mt": "#ff7f0e",
    "tts": "#2ca02c",
    "fp32": "#888888",
    "int8": "#1f77b4",
    "highlight": "#d62728",
    "neutral": "#4c72b0",
    "alt": "#dd8452",
    "filtered": "#2ca02c",
    "raw": "#d62728",
}


def _load(name: str):
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _ygrid(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.8)
    ax.set_axisbelow(True)


def _footnote(fig, text):
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8, color="#555555")


def fig1_latency_stack():
    d = _load("bench_compare_en_n5.json")
    b = d["baseline"]["aggregate"]
    f = d["fast"]["aggregate"]
    asr = [b["latency_ms_asr"]["mean"], f["latency_ms_asr"]["mean"]]
    mt = [b["latency_ms_mt"]["mean"], f["latency_ms_mt"]["mean"]]
    tts = [b["latency_ms_tts"]["mean"], f["latency_ms_tts"]["mean"]]
    total = [b["latency_ms_total"]["mean"], f["latency_ms_total"]["mean"]]
    labels = ["FP32 baseline", "INT8 fast"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = list(range(len(labels)))
    width = 0.55
    p1 = ax.bar(x, asr, width, label="ASR", color=PALETTE["asr"])
    p2 = ax.bar(x, mt, width, bottom=asr, label="MT", color=PALETTE["mt"])
    bottom_tts = [a + m for a, m in zip(asr, mt)]
    p3 = ax.bar(x, tts, width, bottom=bottom_tts, label="TTS", color=PALETTE["tts"])
    for xi, t in zip(x, total):
        ax.text(xi, t + max(total) * 0.02, f"{t/1000:.1f} s",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Per-stage latency: FP32 vs INT8 cascade (n=5, VIVOS-dev)")
    ax.legend(loc="upper right", frameon=False)
    _ygrid(ax)
    ax.set_ylim(0, max(total) * 1.18)
    _footnote(fig, "Source: Results/bench_compare_en_n5.json — PhoWhisper-small + NLLB-200-distilled-600M, CPU, vi→en")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig1_latency_stack.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_ablation_speedup():
    d = _load("ablation_en_n3.json")
    rows = d["headline_table"]
    ids = [r["config_id"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    latencies = [r["latency_mean_ms"] for r in rows]

    colors = []
    for cid in ids:
        if cid == "full":
            colors.append(PALETTE["highlight"])
        elif cid == "int8_cascade":
            colors.append(PALETTE["fp32"])
        else:
            colors.append(PALETTE["neutral"])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = list(range(len(ids)))
    bars = ax.bar(x, speedups, color=colors, width=0.62)
    for xi, sp, lat in zip(x, speedups, latencies):
        ax.text(xi, sp + 0.03, f"{sp:.2f}x\n({lat/1000:.1f}s)",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("int8_", "") for s in ids], rotation=15, ha="right")
    ax.set_ylabel("Speedup vs INT8 cascade (×)")
    ax.set_title("Component ablation: speedup contribution per cache / tone-VAD layer")
    ax.set_ylim(0, max(speedups) * 1.25)
    _ygrid(ax)
    _footnote(fig, "Source: Results/ablation_en_n3.json — baseline = int8_cascade (1.0×). n=3 utterances; latencies in legend.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig2_ablation_speedup.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_fingerprint_ablation():
    d = _load("fingerprint_ablation.json")
    rows = d["ablations"]
    names = [r["ablation"] for r in rows]
    intra = [r["intra_cos_mean"] for r in rows]
    inter = [r["inter_cos_mean"] for r in rows]
    gap = [r["gap"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                    gridspec_kw={"width_ratios": [1.6, 1.0]})
    x = list(range(len(names)))
    width = 0.38
    ax1.bar([xi - width/2 for xi in x], intra, width, label="intra-class cos",
            color=PALETTE["neutral"])
    ax1.bar([xi + width/2 for xi in x], inter, width, label="inter-class cos",
            color=PALETTE["alt"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Fingerprint component ablation: intra vs inter cosine")
    ax1.legend(loc="lower left", frameon=False)
    ax1.set_ylim(0, 1.05)
    _ygrid(ax1)

    colors2 = [PALETTE["highlight"] if g < 0 else PALETTE["neutral"] for g in gap]
    ax2.bar(x, gap, color=colors2, width=0.6)
    ax2.axhline(0, color="#333333", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Discriminative gap (intra − inter)")
    ax2.set_title("Gap (positive = separable)")
    _ygrid(ax2)

    _footnote(fig, "Source: Results/fingerprint_ablation.json — n_intra=50, n_inter=50 pairs. Red bars = gap≤0 (not separable).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig3_fingerprint_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_minimal_pair_ablation():
    d = _load("minimal_pair_eval.json")
    rows = d["variants"]
    names = [r["variant"] for r in rows]
    intra_tone = [r["intra_tone_cos_mean"] for r in rows]
    inter_syl = [r["inter_syl_cos_mean"] for r in rows]
    protect = [r["protect_score"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                    gridspec_kw={"width_ratios": [1.4, 1.0]})
    x = list(range(len(names)))
    width = 0.38
    ax1.bar([xi - width/2 for xi in x], intra_tone, width,
            label="intra-tone (same syl/diff tone)", color=PALETTE["neutral"])
    ax1.bar([xi + width/2 for xi in x], inter_syl, width,
            label="inter-syl (diff syl/same tone)", color=PALETTE["alt"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("Tonal-minimal-pair protection by fingerprint variant")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_ylim(0, 1.05)
    _ygrid(ax1)

    colors = []
    for n, p in zip(names, protect):
        if n == "tone_only":
            colors.append(PALETTE["highlight"])
        elif p > 0:
            colors.append(PALETTE["filtered"])
        else:
            colors.append(PALETTE["fp32"])
    ax2.bar(x, protect, color=colors, width=0.6)
    ax2.axhline(0, color="#333333", linewidth=0.6)
    for xi, p, n in zip(x, protect, names):
        ax2.text(xi, p + (0.002 if p >= 0 else -0.004),
                 f"{p:+.4f}", ha="center",
                 va="bottom" if p >= 0 else "top", fontsize=9,
                 fontweight="bold" if n == "tone_only" else "normal")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Protect score (inter_syl − intra_tone)")
    ax2.set_title("Protection gap (positive = tone-protective)")
    _ygrid(ax2)

    _footnote(fig, "Source: Results/minimal_pair_eval.json — 6 syllable groups × 6 tones; only tone_only achieves positive protect_score.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig4_minimal_pair_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_noise_robustness():
    d = _load("noise_robustness_eval.json")
    rows = sorted(d["snr_curves"], key=lambda r: r["snr_db_target"])
    snr = [r["snr_db_target"] for r in rows]
    raw = [r["cache_cosine_raw_mean"] for r in rows]
    raw_std = [r["cache_cosine_raw_std"] for r in rows]
    filt = [r["cache_cosine_filtered_mean"] for r in rows]
    filt_std = [r["cache_cosine_filtered_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(snr, raw, yerr=raw_std, marker="o", color=PALETTE["raw"],
                label="raw cache cosine", linewidth=1.8, capsize=3)
    ax.errorbar(snr, filt, yerr=filt_std, marker="s", color=PALETTE["filtered"],
                label="with spectral filter", linewidth=1.8, capsize=3)
    ax.axhline(0.95, color="#444444", linestyle="--", linewidth=1.0,
               label=r"$\tau$ = 0.95 (cache-hit threshold)")
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("Mean cache cosine")
    ax.set_title("Noise robustness of fingerprint cache (DEMAND-augmented VIVOS)")
    ax.set_ylim(-1.05, 1.1)
    ax.legend(loc="lower right", frameon=False)
    ax.invert_xaxis()
    _ygrid(ax)
    _footnote(fig, "Source: Results/noise_robustness_eval.json — n=10 utterances/SNR; spectral filter restores τ=0.95 hits down to ~15 dB.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig5_noise_robustness.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig6_two_stream_voting():
    d = _load("two_stream_voting_eval.json")
    labels = ["self-hit\n(same utt.)", "correct-miss\n(tonal variant)", "false-hit\n(tonal variant)"]
    vals = [d["self_hit_rate"], d["tonal_variant_correct_miss_rate"], d["tonal_variant_false_hit_rate"]]
    colors = [PALETTE["neutral"], PALETTE["filtered"], PALETTE["highlight"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = list(range(len(labels)))
    ax.bar(x, vals, color=colors, width=0.55)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025, f"{v*100:.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    cs = d["cache_stats"]
    ann = (f"cache: {cs['n_entries']} entries, {cs['lookups']} lookups\n"
           f"tone-only hits: {cs['tone_only_hits']}, segment-only hits: {cs['segment_only_hits']}\n"
           f"both-stream hits: {cs['both_hits']}")
    ax.text(0.98, 0.55, ann, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", boxstyle="round,pad=0.4"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.18)
    ax.set_title("Two-stream voting cache: 100% correct-miss on tonal variants, 0% false-hit")
    _ygrid(ax)
    _footnote(fig, "Source: Results/two_stream_voting_eval.json — both segment and tone streams must agree to fire a hit.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig6_two_stream_voting.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig7_worker_scale():
    d = _load("worker_simulation.json")
    scen = d["scenarios"]
    n_workers = [s["config"]["n_workers"] for s in scen]
    hit = [s["t1_hit_rate"] * 100 for s in scen]
    lat_mean = [s["latency_ms"]["mean"] for s in scen]
    lat_p95 = [s["latency_ms"]["p95"] for s in scen]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x = list(range(len(n_workers)))
    width = 0.55
    bars = ax1.bar(x, hit, width, color=PALETTE["neutral"], label="T1 hit rate (%)")
    for xi, h in zip(x, hit):
        ax1.text(xi, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w} worker{'s' if w > 1 else ''}" for w in n_workers])
    ax1.set_ylabel("T1 cache hit rate (%)", color=PALETTE["neutral"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["neutral"])
    ax1.set_ylim(0, max(hit) * 1.35)
    _ygrid(ax1)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, lat_mean, marker="o", color=PALETTE["highlight"],
             linewidth=2.0, label="mean latency (ms)")
    ax2.plot(x, lat_p95, marker="s", linestyle="--", color=PALETTE["alt"],
             linewidth=1.6, label="p95 latency (ms)")
    ax2.set_ylabel("Latency (ms)", color=PALETTE["highlight"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["highlight"])
    ax2.set_ylim(0, max(lat_p95) * 1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=False)
    ax1.set_title("Worker-scale simulation: T1 hit rate stable across worker counts")
    _footnote(fig, "Source: Results/worker_simulation.json — 2 phrases/worker × 3 repeats; hit rate constant at ~66.7% (= 2/3 repeat hits).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT / "fig7_worker_scale.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    generators = [
        fig1_latency_stack,
        fig2_ablation_speedup,
        fig3_fingerprint_ablation,
        fig4_minimal_pair_ablation,
        fig5_noise_robustness,
        fig6_two_stream_voting,
        fig7_worker_scale,
    ]
    written = []
    for gen in generators:
        try:
            path = gen()
            written.append(path)
            print(f"[OK]  {gen.__name__:32s} -> {path}")
        except Exception as exc:
            print(f"[ERR] {gen.__name__:32s}: {exc!r}")
            raise

    print("\n--- file report ---")
    try:
        from PIL import Image
        have_pil = True
    except ImportError:
        have_pil = False
    for p in written:
        sz = p.stat().st_size
        if have_pil:
            with Image.open(p) as im:
                w, h = im.size
            print(f"{p.name:36s}  {sz/1024:7.1f} KB   {w}x{h} px")
        else:
            print(f"{p.name:36s}  {sz/1024:7.1f} KB   (install Pillow for resolution)")


if __name__ == "__main__":
    main()
