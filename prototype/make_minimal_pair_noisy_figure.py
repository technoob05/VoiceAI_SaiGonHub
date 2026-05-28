"""Render papers/figures/fig_minimal_pair_noisy.png from
Results/minimal_pair_eval_large.json.

Bar chart: x-axis = condition (clean, 20dB, 15dB, 10dB, 5dB),
3 grouped bars per condition = (self-lookup recall, correct-miss rate,
false-hit rate). Reports for tau_A=0.85 tau_B=0.93 (deployment default).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "Results" / "minimal_pair_eval_large.json"
OUT = PROJECT / "papers" / "figures" / "fig_minimal_pair_noisy.png"


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rbc = data["results_by_condition"]
    order = ["clean", "snr_20dB", "snr_15dB", "snr_10dB", "snr_5dB"]
    labels = ["Clean", "20 dB", "15 dB", "10 dB", "5 dB"]
    op_key = "tau_A=0.85_tau_B=0.93"

    self_recall    = []
    correct_miss   = []
    false_hit      = []
    for c in order:
        r = rbc[c]["by_operating_point"][op_key]
        self_recall.append(r["self_recall"])
        correct_miss.append(r["correct_miss_rate"])
        false_hit.append(r["false_hit_rate"])

    x = np.arange(len(order))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    b1 = ax.bar(x - width, self_recall,  width, label="self-lookup recall", color="#2ca02c")
    b2 = ax.bar(x,         correct_miss, width, label="correct-miss rate",  color="#1f77b4")
    b3 = ax.bar(x + width, false_hit,    width, label="false-hit rate",    color="#d62728")

    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + 0.012, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (proportion)")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Existence-rule two-stream voting on 120 minimal-pair clips "
                 f"(20 syllables x 6 tones), tau_A=0.85 tau_B=0.93")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
