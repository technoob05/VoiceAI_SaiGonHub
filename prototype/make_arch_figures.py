"""Architecture diagram generator for VoiceAI_SaiGonHub paper.

Produces three paper-quality vector PDFs (and PNG twins) using only
matplotlib.patches primitives. Pure-Python, CPU-only, runs on Windows.

Outputs (in papers/figures/):
    fig0_pipeline.pdf / .png            — end-to-end system pipeline
    fig8_two_stream_voting_arch.pdf/.png — two-stream voting AND-gate
    fig9_fingerprint_composition.pdf/.png — 272-d fingerprint breakdown
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

# Soft pastels
C = {
    "mic":      "#E0E0E0",
    "vad":      "#FFE5B4",
    "cache_a":  "#C8E6C9",
    "cache_b":  "#A5D6A7",
    "miss":     "#FFCCBC",
    "asr":      "#BBDEFB",
    "lm":       "#B3E5FC",
    "mt":       "#D1C4E9",
    "tts":      "#F8BBD0",
    "speaker":  "#CFD8DC",
    "tone":     "#FFF59D",
    "seg":      "#B2DFDB",
    "and":      "#FFAB91",
    "hit":      "#A5D6A7",
    "delta":    "#80CBC4",
    "f0":       "#CE93D8",
    "energy":   "#FFAB91",
}

BORDER = "black"
LW = 1.2


def _box(ax, x, y, w, h, color, text, fontsize=9, weight="normal", rounded=True):
    if rounded:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=LW, edgecolor=BORDER, facecolor=color,
        )
    else:
        patch = Rectangle(
            (x, y), w, h, linewidth=LW, edgecolor=BORDER, facecolor=color,
        )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color="black", lw=1.2, style="->",
           connectionstyle="arc3,rad=0", label=None, label_pos=None,
           label_color=None, label_fontsize=8):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
    )
    ax.add_patch(arr)
    if label is not None:
        lx, ly = label_pos if label_pos else ((x1 + x2) / 2, (y1 + y2) / 2)
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=label_fontsize,
                color=label_color or color,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.9))


# ---------------------------------------------------------------------------
# Figure 0 — pipeline
# ---------------------------------------------------------------------------
def make_pipeline():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top row (cache-hit fast path) — y ≈ 5.7
    _box(ax, 0.1, 5.7, 1.2, 0.9, C["mic"], "Mic", fontsize=9)
    _box(ax, 1.6, 5.7, 1.7, 0.9, C["vad"], "Tone-VAD\n§3.7", fontsize=8)
    _box(ax, 3.6, 5.7, 1.9, 0.9, C["cache_a"], "T1 audio cache\n§3.3", fontsize=8)
    _box(ax, 6.0, 5.7, 1.7, 0.9, C["cache_b"], "T2 audio out\n(cached)", fontsize=8)
    _box(ax, 11.6, 5.7, 1.3, 0.9, C["speaker"], "Speaker", fontsize=9)

    # Middle row (miss path: ASR + LM rerank) — y ≈ 3.7
    _box(ax, 3.6, 3.7, 1.9, 0.9, C["asr"], "INT8 ASR\n§3.8", fontsize=8)
    _box(ax, 6.0, 3.7, 2.1, 0.9, C["lm"], "3-gram LM rerank\n§3.9", fontsize=8)
    _box(ax, 8.6, 3.7, 1.9, 0.9, C["cache_a"], "T2 text cache\n§3.4", fontsize=8)

    # Lower row (full miss: NLLB -> TTS) — y ≈ 1.5
    _box(ax, 3.6, 1.5, 1.9, 0.9, C["mt"], "INT8 NLLB-200", fontsize=8)
    _box(ax, 6.0, 1.5, 1.7, 0.9, C["tts"], "INT8 TTS", fontsize=8)

    # T2-hit branch TTS — slightly above lower row to receive vertical arrow
    _box(ax, 8.6, 1.5, 1.7, 0.9, C["tts"], "INT8 TTS", fontsize=8)
    _box(ax, 11.6, 1.5, 1.3, 0.9, C["speaker"], "Speaker", fontsize=9)

    # === Arrows ===
    # Top row Mic -> VAD -> T1
    _arrow(ax, 1.3, 6.15, 1.6, 6.15)
    _arrow(ax, 3.3, 6.15, 3.6, 6.15)
    # T1 hit -> T2 audio out (green)
    _arrow(ax, 5.5, 6.15, 6.0, 6.15, color="#2E7D32", lw=1.6,
           label="T1 hit · 1441× speedup", label_pos=(5.78, 6.85),
           label_color="#2E7D32")
    # T2 audio out -> Speaker
    _arrow(ax, 7.7, 6.15, 11.6, 6.15, color="#2E7D32", lw=1.6)

    # T1 miss -> ASR (orange)
    _arrow(ax, 4.55, 5.7, 4.55, 4.6, color="#E65100", lw=1.4,
           label="miss", label_pos=(5.05, 5.15),
           label_color="#E65100")

    # ASR -> LM rerank -> T2 text cache (black, internal)
    _arrow(ax, 5.5, 4.15, 6.0, 4.15)
    _arrow(ax, 8.1, 4.15, 8.6, 4.15)

    # T2 hit -> INT8 TTS (green vertical down)
    _arrow(ax, 9.55, 3.7, 9.55, 2.4, color="#2E7D32", lw=1.6,
           label="T2 hit · ~3-12× skip", label_pos=(10.55, 3.05),
           label_color="#2E7D32")

    # T2 miss -> NLLB. Route the horizontal "miss" leg BELOW the row of
    # boxes so it does not visually cross through 3-gram LM rerank.
    # Drop down from T2-text bottom, traverse left at y≈3.0, then down to NLLB.
    _arrow(ax, 9.55, 3.7, 9.55, 3.0, color="#E65100", lw=1.4)
    _arrow(ax, 9.55, 3.0, 4.55, 3.0, color="#E65100", lw=1.4,
           label="miss", label_pos=(6.0, 3.22),
           label_color="#E65100")
    _arrow(ax, 4.55, 3.0, 4.55, 2.4, color="#E65100", lw=1.4)

    # NLLB -> TTS (bottom) -> Speaker
    _arrow(ax, 5.5, 1.95, 6.0, 1.95, color="#E65100", lw=1.4)
    _arrow(ax, 7.7, 1.95, 11.6, 1.95, color="#E65100", lw=1.4)

    # Legend (bottom-left)
    ax.text(0.1, 0.65, "Green = cache-hit fast path",
            color="#2E7D32", fontsize=8, weight="bold")
    ax.text(0.1, 0.25, "Orange = miss / full-cascade path",
            color="#E65100", fontsize=8, weight="bold")
    # Annotation (bottom-right)
    ax.text(13, 0.45,
            "INT8 cascade total: 6.7 s\nT1 hit: 25 ms (1441× over FP32 36.5 s)",
            ha="right", va="bottom", fontsize=7.5, style="italic",
            color="#333333")

    plt.tight_layout()
    fig.savefig(OUT / "fig0_pipeline.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig0_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — two-stream voting
# ---------------------------------------------------------------------------
def make_two_stream():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Inset placed at top (full width above streams, no overlap)
    inset = FancyBboxPatch(
        (0.2, 7.6), 5.6, 1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.0, edgecolor="#888888", facecolor="#FFF3E0",
    )
    ax.add_patch(inset)
    ax.text(3.0, 8.45, "Tonal minimal pair: ma vs má",
            ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(3.0, 7.95,
            "stream A disagrees → vetoed, even though stream B says hit",
            ha="center", va="center", fontsize=7.5,
            color="#444444", style="italic")

    # Query (centered vertically between the two streams)
    _box(ax, 0.2, 3.5, 2.0, 1.0, C["mic"], "Query audio  q", fontsize=9)

    # Stream A — tone (top)
    _box(ax, 2.9, 5.2, 4.4, 1.5, C["tone"],
         "Stream A — tone\n$f_A$ : 6-d (tone histogram)\n"
         r"$\mathrm{argmax}_e \cos(f_A(q), f_A(e))$",
         fontsize=8)
    _box(ax, 7.7, 5.5, 1.4, 0.9, "#FFF8E1",
         r"$\tau_A = 0.85$", fontsize=9, weight="bold")

    # Stream B — segmental (bottom)
    _box(ax, 2.9, 1.3, 4.4, 1.5, C["seg"],
         "Stream B — segmental\n$f_B$ : 260-d (MFCC + Δ + F0 + E)\n"
         r"$\mathrm{argmax}_e \cos(f_B(q), f_B(e))$",
         fontsize=8)
    _box(ax, 7.7, 1.6, 1.4, 0.9, "#FFF8E1",
         r"$\tau_B = 0.93$", fontsize=9, weight="bold")

    # AND gate
    _box(ax, 9.5, 3.5, 1.2, 1.4, C["and"], "AND", fontsize=11, weight="bold")
    # HIT
    _box(ax, 11.1, 3.8, 0.8, 0.8, C["hit"], "HIT", fontsize=10, weight="bold")

    # Arrows from query to streams
    _arrow(ax, 2.2, 4.3, 2.9, 6.0)
    _arrow(ax, 2.2, 3.7, 2.9, 2.0)

    # Stream -> threshold
    _arrow(ax, 7.3, 5.95, 7.7, 5.95)
    _arrow(ax, 7.3, 2.05, 7.7, 2.05)

    # Threshold -> AND (gray)
    _arrow(ax, 9.1, 5.95, 9.8, 4.9, color="#444444")
    _arrow(ax, 9.1, 2.05, 9.8, 3.5, color="#444444")

    # AND -> HIT (green)
    _arrow(ax, 10.7, 4.2, 11.1, 4.2, lw=1.6, color="#2E7D32")

    # Annotation under streams
    ax.text(6.0, 0.4,
            "AND gate fires iff both streams nominate the SAME stored entry id "
            r"and $\cos_A \geq \tau_A$ and $\cos_B \geq \tau_B$",
            ha="center", va="center", fontsize=8, style="italic",
            color="#333333")

    plt.tight_layout()
    fig.savefig(OUT / "fig8_two_stream_voting_arch.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig8_two_stream_voting_arch.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — 272-d fingerprint composition
# ---------------------------------------------------------------------------
def make_fingerprint_composition():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Total = 272; widths use a hybrid scheme — boxes mostly proportional but
    # with a strong floor so even tiny components (4-d F0, 2-d Energy) get a
    # readable box; the actual dim count is the load-bearing text.
    parts = [
        ("Tone hist",     6,   C["tone"],   "Vi 6 tones"),
        ("Temporal MFCC", 130, C["asr"],    "10×13 MFCC chunks"),
        ("MFCC Δ",        130, C["delta"],  "10×13 Δ-chunks"),
        ("F0 stats",      4,   C["f0"],     "mean, std,\nrange, slope"),
        ("Energy",        2,   C["energy"], "log-RMS\nmean, peak"),
    ]

    bar_y = 4.3
    bar_h = 1.6
    bar_x0 = 0.5
    bar_x1 = 13.5
    bar_w = bar_x1 - bar_x0

    # Use sqrt(dims) so small bars are visible, then renormalize to bar_w.
    import math
    weights = [math.sqrt(p[1]) for p in parts]
    s = sum(weights)
    widths = [bar_w * w / s for w in weights]
    # Apply a minimum width so the smallest boxes still hold their label.
    min_w = 1.9
    over = sum(max(0, min_w - w) for w in widths)
    big_total = sum(w for w in widths if w >= min_w)
    widths = [
        max(min_w, w) if w < min_w
        else w - over * (w / big_total)
        for w in widths
    ]

    x = bar_x0
    for (name, dims, color, ann), w in zip(parts, widths):
        rect = Rectangle((x, bar_y), w, bar_h,
                         linewidth=LW, edgecolor=BORDER, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, bar_y + bar_h * 0.65, name,
                ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, bar_y + bar_h * 0.28, f"d = {dims}",
                ha="center", va="center", fontsize=8)
        ax.text(x + w / 2, bar_y - 0.55, ann,
                ha="center", va="top", fontsize=7.5, color="#333333")
        ax.plot([x, x + w], [bar_y - 0.12, bar_y - 0.12],
                color="#777777", lw=0.8)
        x += w

    # Σ annotation
    ax.text(bar_x1 + 0.4, bar_y + bar_h / 2, r"$\Sigma = 272$",
            ha="left", va="center", fontsize=12, weight="bold",
            color="#222222")

    # Title
    ax.text(7.5, 7.4, "Tonal-perceptual fingerprint composition",
            ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(7.5, 6.85,
            "Five concatenated, L2-normalized components → 272-d unit vector",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#444444")

    # Footnote — two-stream split
    ax.text(7.5, 1.7,
            "Two-stream voting (§3.5) splits this into independent streams:\n"
            r"$f_A$ = component A (6-d, tone)     $\perp$     "
            r"$f_B$ = components B+C+D+E (260-d, segmental)",
            ha="center", va="center", fontsize=8.5, color="#222222")

    plt.tight_layout()
    fig.savefig(OUT / "fig9_fingerprint_composition.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig9_fingerprint_composition.png", bbox_inches="tight")
    plt.close(fig)


def main():
    make_pipeline()
    make_two_stream()
    make_fingerprint_composition()
    for stem in ["fig0_pipeline", "fig8_two_stream_voting_arch",
                 "fig9_fingerprint_composition"]:
        for ext in ["pdf", "png"]:
            p = OUT / f"{stem}.{ext}"
            sz = p.stat().st_size if p.exists() else -1
            print(f"  {p.name:48s} {sz:>8d} bytes")


if __name__ == "__main__":
    main()
