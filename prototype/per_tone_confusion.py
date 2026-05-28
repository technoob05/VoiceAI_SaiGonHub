"""
Per-tone confusion analysis on the 120-clip Vietnamese minimal-pair corpus.

Question: WHICH Vietnamese tones does the existence-rule two-stream voting
cache confuse with which, under (a) the heuristic 6-d tone histogram and
(b) the learned 32-d tone embedding?

Protocol (mirrors minimal_pair_eval_large.py / learned_tone_stream.py):
  - Re-synthesize the 20 carrier syllables x 6 tones = 120 clips with
    facebook/mms-tts-vie (clean only).
  - For each "inserted" tone T in {0..5}:
      build a fresh TwoStreamCache; insert all 20 syllables in tone T;
      probe with each of the 20 syllables in EVERY tone T' in {0..5};
      count cache hits.
  - Cell C[T, T'] = (#hits) / 20 (fraction of probes that hit). Diagonal
    cells are self-cache-hits (sanity); off-diagonal cells are tonal
    confusions.

Operating points:
  - Heuristic (6-d tone hist) at (tau_A, tau_B) = (0.85, 0.93)
  - Learned (32-d embedding)  at (tau_A, tau_B) = (0.90, 0.93)

Outputs:
  - Results/per_tone_confusion.json
  - papers/figures/fig_per_tone_confusion.png  (two heatmaps side-by-side)

Pure NumPy + matplotlib. No GPU. Reuses learned weights at
data/cache_snapshots/learned_tone_stream.npz if loadable; otherwise
the learned condition is skipped honestly.
"""
from __future__ import annotations

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
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                                  # noqa: E402
import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from two_stream_voting import (                                     # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
)


CARRIER = "noi {syl}"

# 20 carrier syllables x 6 tones in fixed order:
#   0 ngang, 1 sac, 2 hoi, 3 nang, 4 nga, 5 huyen
TONE_NAMES = ["ngang", "sắc", "hỏi", "nặng", "ngã", "huyền"]
TONE_NAMES_ASCII = ["ngang", "sac", "hoi", "nang", "nga", "huyen"]

SYLLABLES_20 = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
    "ta":  ["ta",  "tá",  "tả",  "tạ",  "tã",  "tà"],
    "na":  ["na",  "ná",  "nả",  "nạ",  "nã",  "nà"],
    "sa":  ["sa",  "sá",  "sả",  "sạ",  "sã",  "sà"],
    "ha":  ["ha",  "há",  "hả",  "hạ",  "hã",  "hà"],
    "va":  ["va",  "vá",  "vả",  "vạ",  "vã",  "và"],
    "pa":  ["pa",  "pá",  "pả",  "pạ",  "pã",  "pà"],
    "ga":  ["ga",  "gá",  "gả",  "gạ",  "gã",  "gà"],
    "ka":  ["ka",  "ká",  "kả",  "kạ",  "kã",  "kà"],
    "ra":  ["ra",  "rá",  "rả",  "rạ",  "rã",  "rà"],
    "xa":  ["xa",  "xá",  "xả",  "xạ",  "xã",  "xà"],
    "gia": ["gia", "giá", "giả", "giạ", "giã", "già"],
    "kha": ["kha", "khá", "khả", "khạ", "khã", "khà"],
    "nga": ["nga", "ngá", "ngả", "ngạ", "ngã", "ngà"],
    "nha": ["nha", "nhá", "nhả", "nhạ", "nhã", "nhà"],
    "tha": ["tha", "thá", "thả", "thạ", "thã", "thà"],
}


def _synth_corpus():
    """Synthesize the 120 clean clips. Returns dict (base, tone_idx) -> audio16k."""
    try:
        from transformers import AutoTokenizer, VitsModel
        import torch
    except Exception as e:
        print(f"[abort] cannot import transformers/torch: {e}")
        return {}, 16000
    print("[load] MMS-TTS-vie ...")
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    out = {}
    t0 = time.perf_counter()
    for base, variants in SYLLABLES_20.items():
        for tone_idx, v in enumerate(variants):
            torch.manual_seed(tone_idx * 7 + (hash(base) & 0xFFFF))
            inputs = tok(CARRIER.format(syl=v), return_tensors="pt")
            with torch.inference_mode():
                wav = model(**inputs).waveform
            a = wav.squeeze().cpu().numpy().astype(np.float32)
            if sr != 16000:
                n16 = int(round(len(a) * 16000 / sr))
                a = np.interp(np.linspace(0, len(a), n16, endpoint=False),
                              np.arange(len(a)), a).astype(np.float32)
            out[(base, tone_idx)] = a
    print(f"[synth] {len(out)} clips in {time.perf_counter()-t0:.1f}s")
    return out, 16000


def _build_confusion(cache_factory, audios, tau_A: float, tau_B: float) -> np.ndarray:
    """For each inserted tone T, count hits per probe tone T'.

    Returns 6x6 numpy array C[T, T'] of hit fractions in [0, 1].
    """
    n_tones = 6
    n_syl = len(SYLLABLES_20)
    C = np.zeros((n_tones, n_tones), dtype=np.float64)
    for T in range(n_tones):
        cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B,
                              max_entries=1024)
        cache = cache_factory(cfg)
        # Insert all 20 syllables in tone T
        for base, variants in SYLLABLES_20.items():
            key = (base, T)
            if key not in audios:
                continue
            fp = cache.fingerprint(audios[key], 16000)
            cache.put(fp, "en",
                      {"vi": f"{base}/{variants[T]}", "tgt": variants[T],
                       "insert_tone": T, "base": base})
        # Probe with every (base, T') for T' in 0..5
        for T_probe in range(n_tones):
            hits = 0
            n_probes = 0
            for base in SYLLABLES_20:
                key = (base, T_probe)
                if key not in audios:
                    continue
                n_probes += 1
                fp = cache.fingerprint(audios[key], 16000)
                h = cache.lookup_existence(fp, "en")
                if h is not None:
                    hits += 1
            C[T, T_probe] = hits / max(n_probes, 1)
    return C


def _try_load_learned():
    """Try to instantiate the learned-tone two-stream cache. Returns
    (factory_callable, tau_A, info) or (None, None, error_msg)."""
    try:
        from learned_tone_stream import (
            ToneEmbeddingMLP,
            LearnedToneTwoStreamCache,
        )
    except Exception as e:
        return None, None, f"import_failed: {e}"
    weights = PROJECT / "data" / "cache_snapshots" / "learned_tone_stream.npz"
    if not weights.exists():
        return None, None, f"weights_missing: {weights}"
    try:
        mlp = ToneEmbeddingMLP.load(weights)
    except Exception as e:
        return None, None, f"load_failed: {e}"
    def factory(cfg):
        return LearnedToneTwoStreamCache(mlp, cfg)
    info = {"weights": str(weights.relative_to(PROJECT)),
            "in_dim": int(mlp.W1.shape[0]),
            "hidden": int(mlp.W1.shape[1]),
            "out_dim": int(mlp.W2.shape[1])}
    return factory, 0.90, info


def _plot_heatmaps(C_heur: np.ndarray, C_learn, fig_path: Path,
                   learn_label: str):
    n_panels = 2 if C_learn is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 5.4), dpi=150)
    if n_panels == 1:
        axes = [axes]

    def _draw(ax, C, title):
        im = ax.imshow(C, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(TONE_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(TONE_NAMES)
        ax.set_xlabel("Probe tone $T'$")
        ax.set_ylabel("Inserted tone $T$")
        ax.set_title(title)
        for i in range(6):
            for j in range(6):
                val = C[i, j]
                # Pick text colour for contrast
                txt_color = "white" if val < 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=9)
        return im

    im0 = _draw(axes[0], C_heur,
                "Heuristic 6-d tone hist\n$(\\tau_A=0.85,\\tau_B=0.93)$")
    if C_learn is not None:
        im1 = _draw(axes[1], C_learn,
                    f"Learned 32-d tone emb\n{learn_label}")
        fig.colorbar(im1, ax=axes, shrink=0.78, label="Cache-hit fraction")
    else:
        fig.colorbar(im0, ax=axes, shrink=0.78, label="Cache-hit fraction")
    fig.suptitle("Per-tone confusion: existence-rule two-stream voting cache",
                 fontsize=12)
    fig.savefig(str(fig_path), bbox_inches="tight")
    plt.close(fig)


def _summarize(C: np.ndarray, label: str) -> dict:
    """Most-confused off-diagonal pair, least-confused, easiest-to-discriminate
    tone (lowest row-sum of off-diagonal hits)."""
    n = C.shape[0]
    off = C.copy()
    np.fill_diagonal(off, np.nan)
    # argmax over off-diagonal
    flat = np.nan_to_num(off, nan=-1.0)
    i_max, j_max = np.unravel_index(np.argmax(flat), flat.shape)
    # smallest off-diagonal value (most discriminable pair)
    flat_min = np.where(np.isnan(off), np.inf, off)
    i_min, j_min = np.unravel_index(np.argmin(flat_min), flat_min.shape)
    # row sums of off-diagonal: how much insert_tone i is confused with anything else
    row_off_mean = np.nanmean(off, axis=1)
    easiest_insert = int(np.argmin(row_off_mean))
    hardest_insert = int(np.argmax(row_off_mean))
    return {
        "label": label,
        "matrix": C.tolist(),
        "tone_names": TONE_NAMES,
        "diag_mean": float(np.mean(np.diag(C))),
        "off_diag_mean": float(np.nanmean(off)),
        "most_confused_pair": {
            "insert_tone": TONE_NAMES[i_max],
            "probe_tone":  TONE_NAMES[j_max],
            "insert_tone_idx": int(i_max),
            "probe_tone_idx":  int(j_max),
            "hit_fraction":    float(C[i_max, j_max]),
        },
        "least_confused_pair": {
            "insert_tone": TONE_NAMES[i_min],
            "probe_tone":  TONE_NAMES[j_min],
            "insert_tone_idx": int(i_min),
            "probe_tone_idx":  int(j_min),
            "hit_fraction":    float(C[i_min, j_min]),
        },
        "easiest_to_discriminate_insert_tone": {
            "tone": TONE_NAMES[easiest_insert],
            "tone_idx": easiest_insert,
            "mean_off_diag_hit": float(row_off_mean[easiest_insert]),
        },
        "hardest_to_discriminate_insert_tone": {
            "tone": TONE_NAMES[hardest_insert],
            "tone_idx": hardest_insert,
            "mean_off_diag_hit": float(row_off_mean[hardest_insert]),
        },
    }


def main() -> int:
    out_json = PROJECT / "Results" / "per_tone_confusion.json"
    out_fig  = PROJECT / "papers" / "figures" / "fig_per_tone_confusion.png"

    # 1. Synthesize the 120-clip corpus
    audios, sr = _synth_corpus()
    if len(audios) < 120:
        print(f"[abort] only synthesized {len(audios)}/120 clips")
        return 1

    # 2. Heuristic confusion matrix
    print("\n[heuristic] building 6x6 confusion at (tau_A=0.85, tau_B=0.93)")
    t0 = time.perf_counter()
    C_heur = _build_confusion(lambda cfg: TwoStreamCache(cfg),
                              audios, tau_A=0.85, tau_B=0.93)
    print(f"[heuristic] done in {time.perf_counter()-t0:.1f}s")
    print(C_heur.round(2))

    # 3. Learned confusion matrix (skip if weights not loadable)
    learn_factory, learn_tau_A, learn_info = _try_load_learned()
    if learn_factory is not None:
        print(f"\n[learned] using {learn_info}")
        print(f"[learned] building 6x6 confusion at (tau_A={learn_tau_A}, tau_B=0.93)")
        t0 = time.perf_counter()
        C_learn = _build_confusion(learn_factory, audios,
                                   tau_A=learn_tau_A, tau_B=0.93)
        print(f"[learned] done in {time.perf_counter()-t0:.1f}s")
        print(C_learn.round(2))
        learn_label_for_fig = f"$(\\tau_A={learn_tau_A},\\tau_B=0.93)$"
    else:
        C_learn = None
        print(f"\n[learned] skipped: {learn_info}")
        learn_label_for_fig = ""

    # 4. Figure
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    _plot_heatmaps(C_heur, C_learn, out_fig, learn_label_for_fig)
    print(f"\n[fig] -> {out_fig}")

    # 5. JSON
    payload = {
        "n_syllables": 20,
        "n_tones": 6,
        "n_clips": len(audios),
        "tone_names": TONE_NAMES,
        "tone_names_ascii": TONE_NAMES_ASCII,
        "tone_index_order": [
            {"idx": 0, "name": "ngang",  "diacritic": "(none)",  "description": "level"},
            {"idx": 1, "name": "sắc",    "diacritic": "acute",   "description": "high-rising"},
            {"idx": 2, "name": "hỏi",    "diacritic": "hook",    "description": "dipping"},
            {"idx": 3, "name": "nặng",   "diacritic": "dot below","description": "low-glottal"},
            {"idx": 4, "name": "ngã",    "diacritic": "tilde",   "description": "creaky-rising"},
            {"idx": 5, "name": "huyền",  "diacritic": "grave",   "description": "low-falling"},
        ],
        "heuristic": {
            "tau_A": 0.85, "tau_B": 0.93,
            **_summarize(C_heur, "heuristic"),
        },
        "learned": (
            {"tau_A": learn_tau_A, "tau_B": 0.93,
             "weights_info": learn_info,
             **_summarize(C_learn, "learned")}
            if C_learn is not None else
            {"status": "skipped", "reason": learn_info}
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[json] -> {out_json}")

    # 6. Report
    print("\n========= REPORT =========")
    h = payload["heuristic"]
    print(f"Heuristic most-confused pair: insert={h['most_confused_pair']['insert_tone']} "
          f"-> probe={h['most_confused_pair']['probe_tone']}  "
          f"hit_frac={h['most_confused_pair']['hit_fraction']:.2f}")
    print(f"Heuristic easiest-to-discriminate insert tone: "
          f"{h['easiest_to_discriminate_insert_tone']['tone']}  "
          f"mean_off_diag={h['easiest_to_discriminate_insert_tone']['mean_off_diag_hit']:.2f}")
    if C_learn is not None:
        l = payload["learned"]
        print(f"Learned most-confused pair:   insert={l['most_confused_pair']['insert_tone']} "
              f"-> probe={l['most_confused_pair']['probe_tone']}  "
              f"hit_frac={l['most_confused_pair']['hit_fraction']:.2f}")
        print(f"Learned easiest-to-discriminate insert tone: "
              f"{l['easiest_to_discriminate_insert_tone']['tone']}  "
              f"mean_off_diag={l['easiest_to_discriminate_insert_tone']['mean_off_diag_hit']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
