"""
exp04 — Tone-bearing-layer probe on Whisper / PhoWhisper encoder

GO/NO-GO gate for the method spine (tone-aware mixed-precision quantization).
Hypothesis: tone information is concentrated in a small subset of encoder
layers (de la Fuente & Jurafsky, Interspeech 2024 — Mandarin tone on wav2vec2 /
HuBERT / WavLM peaks at layer ~8 of 12). We must replicate that finding on the
*Whisper* family (supervised, multilingual, log-Mel input — different architecture).

Procedure (per encoder layer L):
  1. Forward each Vi utterance through PhoWhisper-small encoder, capture per-layer
     hidden states (T_frames, D).
  2. Tokenize the reference text into syllables; for each syllable extract the
     6-class tone label from its Unicode diacritic.
  3. Split encoder frames uniformly into N segments where N = #syllables,
     mean-pool each segment → 1 (D,)-vector per syllable.
  4. Pool across all utterances → (M, D) features + (M,) tone labels, where
     M = total syllables across the dataset.
  5. Train a L2-regularized logistic-regression probe (5-fold CV) for 6-class
     tone classification. Report macro-F1 and accuracy.

PASS criteria (any one breaks the spine):
  - Tone-encoding must be concentrated in <= 30% of layers (target peak).
  - Peak layer accuracy must beat random (1/6 = 16.7%) by >= 20pp absolute.
  - Mean tone-classification accuracy across BEST 30% of layers must beat
    accuracy across the WORST 30% by >= 10pp.

If those hold, the selective-quantization story is empirically defensible.
If not, pivot to backup spine (see CLAUDE.md §4).

Runs on CPU (slow ~15-30 min for 500 utterances) or Kaggle T4 (~3-5 min).

Self-contained: copy-paste to Kaggle, attach voiceai-tier1-vi dataset, run.

Output: Results/exp04_whisper_tone_probe.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Force UTF-8 stdout (Windows cp1252 fix).
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

# Redirect HF caches to project D: drive (skipped if running on Kaggle).
PROJECT = Path(__file__).resolve().parents[1]
_DEFAULT_HF = PROJECT / "data" / "raw" / "hf_cache"
if _DEFAULT_HF.exists():
    os.environ.setdefault("HF_HOME",      str(_DEFAULT_HF))
    os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF / "hub"))


# ============================================================ Config

@dataclass
class Config:
    EXP_ID:        str = "exp04"
    MODEL_ID:      str = "vinai/PhoWhisper-small"
    # Where to find extracted VIVOS. Kaggle: /kaggle/input/voiceai-tier1-vi/vivos
    VIVOS_ROOT:    str = "data/processed/vivos"
    # Use the larger TRAIN split for probe training data (more syllables).
    SPLIT:         str = "train"
    MAX_UTTERANCES: int = 500          # 500 utterances ~= 4-5K syllables; plenty for probing
    DEVICE:        str = "auto"         # "auto" / "cpu" / "cuda" / "mps"
    DTYPE:         str = "auto"
    SEED:          int = 42
    PROBE_C:       float = 1.0          # L2 regularization
    N_SPLITS:      int = 5              # CV folds
    OUTPUT_DIR:    str = "Results"

    PASS_LAYER_FRAC:    float = 0.30    # tone must concentrate in <= 30% of layers
    PASS_MIN_PEAK_ACC:  float = 0.367   # 1/6 + 20pp
    PASS_BEST_WORST_GAP: float = 0.10   # best 30% layers vs worst 30% gap >= 10pp


# ============================================================ Vietnamese tone extraction
# Same encoding as exp00 — kept inline for self-containment.

_TONE_MARKS: dict[str, int] = {"̀": 1, "́": 2, "̉": 3, "̃": 4, "̣": 5}
TONE_NAMES = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]


def syllable_tone(syllable: str) -> int:
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in _TONE_MARKS:
            return _TONE_MARKS[ch]
    return 0


_PUNCT_RE = re.compile(r"[^\w\s\xC0-ỹ]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")

def normalize_vi(text: str) -> list[str]:
    if text is None: return []
    t = _PUNCT_RE.sub(" ", text.lower())
    t = _WS_RE.sub(" ", t).strip()
    return [s for s in t.split(" ") if s]


# ============================================================ VIVOS local loader

def load_vivos_split(root: str, split: str) -> list[tuple[Path, str]]:
    base = Path(root)
    candidates = [base / "vivos" / split, base / split, base / "vivos" / "vivos" / split]
    cand = next((c for c in candidates if (c / "prompts.txt").exists()), None)
    if cand is None:
        raise FileNotFoundError(f"VIVOS {split} not found under: {candidates}")
    prompts = []
    for line in (cand / "prompts.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        utt_id, _, text = line.partition(" ")
        spk = utt_id.split("_")[0]
        wav = cand / "waves" / spk / f"{utt_id}.wav"
        if wav.exists():
            prompts.append((wav, text.strip()))
    print(f"[data] VIVOS local: {cand}  n_utterances={len(prompts)}")
    return prompts


# ============================================================ Encoder forward + frame->syllable pooling

def extract_layer_features(audio, sr, model, processor, device, dtype):
    """Return list[ np.ndarray(T_frames, D) ] for each encoder layer (output of layer L)."""
    import numpy as np
    import torch

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    feats = inputs["input_features"].to(device, dtype)
    with torch.inference_mode():
        encoder = model.model.encoder if hasattr(model, "model") else model.encoder
        outs = encoder(feats, output_hidden_states=True, return_dict=True)
    # outs.hidden_states is a tuple of length (n_layers + 1):
    #   [embed_output, layer1_out, layer2_out, ..., layerN_out]
    # Each is (B=1, T_frames, D).
    return [h.squeeze(0).float().cpu().numpy() for h in outs.hidden_states]


def syllable_features_from_layers(layer_acts: list, n_syllables: int):
    """Uniformly split T_frames into n_syllables segments per layer, mean-pool each.
    Returns: list of arrays (n_syllables, D), one per layer.
    """
    import numpy as np
    if n_syllables <= 0:
        return []
    T = layer_acts[0].shape[0]
    # Half-open [start, end) boundaries.
    bounds = np.linspace(0, T, n_syllables + 1, dtype=int)
    bounds[-1] = max(bounds[-1], bounds[-2] + 1)  # guarantee >=1 frame per syllable
    out = []
    for L_act in layer_acts:
        pooled = np.stack([L_act[bounds[i]:bounds[i+1]].mean(axis=0)
                           for i in range(n_syllables)], axis=0)
        out.append(pooled)
    return out


# ============================================================ Main

def main() -> int:
    import numpy as np
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    import soundfile as sf

    cfg = Config()
    # Allow env override of MAX_UTTERANCES for fast smoke run
    if os.environ.get("VOICEAI_MAX_UTTERANCES"):
        cfg.MAX_UTTERANCES = int(os.environ["VOICEAI_MAX_UTTERANCES"])

    # Resolve device + dtype
    if cfg.DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.DEVICE
    if cfg.DTYPE == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.DTYPE]

    print(f"[init] device={device} dtype={dtype}")

    # ----- load model + processor (encoder only is enough but loading full is simpler) -----
    print(f"[load] {cfg.MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(cfg.MODEL_ID)
    model     = WhisperForConditionalGeneration.from_pretrained(cfg.MODEL_ID, torch_dtype=dtype).to(device).eval()
    n_layers  = model.config.encoder_layers
    print(f"[info] encoder has {n_layers} layers")

    # ----- load VIVOS split -----
    utts = load_vivos_split(cfg.VIVOS_ROOT, cfg.SPLIT)
    rng = np.random.default_rng(cfg.SEED)
    if cfg.MAX_UTTERANCES and len(utts) > cfg.MAX_UTTERANCES:
        idx = rng.choice(len(utts), cfg.MAX_UTTERANCES, replace=False)
        utts = [utts[i] for i in idx]
    print(f"[plan] {len(utts)} utterances will be encoded")

    # ----- forward pass + per-syllable feature extraction -----
    # Storage: features[layer_idx] = list of (n_syl_i, D) arrays; labels = list of (n_syl_i,) arrays.
    features_per_layer: list[list] = [[] for _ in range(n_layers + 1)]   # +1 = embed output
    labels: list = []

    t_start = time.perf_counter()
    skipped = 0
    for i, (wav, text) in enumerate(utts, start=1):
        syls = normalize_vi(text)
        if not syls:
            skipped += 1; continue
        tone_labels = np.array([syllable_tone(s) for s in syls], dtype=np.int64)
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        try:
            layer_acts = extract_layer_features(audio, sr, model, processor, device, dtype)
            pooled    = syllable_features_from_layers(layer_acts, len(syls))
            if len(pooled) != n_layers + 1:
                # PhoWhisper encoder + embed_output should give n_layers + 1 hidden states
                # but some configs differ; just append what we have
                pass
            for L, arr in enumerate(pooled):
                features_per_layer[L].append(arr)
            labels.append(tone_labels)
        except Exception as e:
            print(f"  [skip] {wav.name}: {type(e).__name__}: {e}")
            skipped += 1

        if i % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [enc] {i}/{len(utts)}  elapsed={elapsed:.0f}s  est_total={elapsed*len(utts)/i:.0f}s  skipped={skipped}")

    print(f"[enc] done in {time.perf_counter()-t_start:.0f}s   skipped={skipped}")

    # Concatenate everything -----
    y = np.concatenate(labels, axis=0)
    print(f"[data] total syllables: M={len(y)}")
    print(f"[data] tone distribution: " + " ".join(
        f"{TONE_NAMES[t]}={int((y==t).sum())}" for t in range(6)
    ))

    # Drop tone classes with too few samples (Stratified K-Fold needs >= n_splits per class).
    class_counts = np.bincount(y, minlength=6)
    valid_classes = np.where(class_counts >= cfg.N_SPLITS)[0]
    if len(valid_classes) < 6:
        print(f"[warn] dropped classes with < {cfg.N_SPLITS} samples: "
              f"{[TONE_NAMES[c] for c in range(6) if c not in valid_classes]}")
    mask = np.isin(y, valid_classes)
    y = y[mask]

    # ----- per-layer probe -----
    layer_results = []
    for L in range(n_layers + 1):
        if not features_per_layer[L]:
            continue
        X = np.concatenate(features_per_layer[L], axis=0)[mask]
        if X.shape[0] != y.shape[0]:
            print(f"  [skip layer {L}] shape mismatch X={X.shape} y={y.shape}")
            continue

        skf = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
        acc_folds, f1_folds = [], []
        for tr_idx, te_idx in skf.split(X, y):
            clf = LogisticRegression(C=cfg.PROBE_C, max_iter=500, multi_class="multinomial",
                                     solver="lbfgs", n_jobs=-1)
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[te_idx])
            acc_folds.append(accuracy_score(y[te_idx], pred))
            f1_folds.append(f1_score(y[te_idx], pred, average="macro"))

        mean_acc = float(np.mean(acc_folds))
        mean_f1  = float(np.mean(f1_folds))
        layer_results.append({"layer": L, "accuracy": round(mean_acc, 4),
                               "macro_f1": round(mean_f1, 4),
                               "accuracy_std": round(float(np.std(acc_folds)), 4)})
        print(f"  [probe] layer {L:>2}: acc={mean_acc:.4f}  macro_f1={mean_f1:.4f}")

    # ----- analyze: peak layer, top-30% layers, gates -----
    accs = np.array([r["accuracy"] for r in layer_results])
    peak_layer = int(layer_results[int(np.argmax(accs))]["layer"])
    peak_acc   = float(accs.max())

    sorted_idx = np.argsort(-accs)                          # descending
    top_k      = max(1, int(round(cfg.PASS_LAYER_FRAC * len(accs))))
    top_layers = [int(layer_results[i]["layer"]) for i in sorted_idx[:top_k]]
    bot_layers = [int(layer_results[i]["layer"]) for i in sorted_idx[-top_k:]]
    top_mean   = float(accs[sorted_idx[:top_k]].mean())
    bot_mean   = float(accs[sorted_idx[-top_k:]].mean())
    best_worst_gap = top_mean - bot_mean

    gates = {
        "peak_acc_beats_random_by_20pp": bool(peak_acc >= cfg.PASS_MIN_PEAK_ACC),
        "best_worst_gap_ge_10pp":        bool(best_worst_gap >= cfg.PASS_BEST_WORST_GAP),
        "tone_concentrated_top30_pct":   None,  # set below
    }

    # Concentration metric: top-30% layers contain at least <50% of total above-baseline mass?
    # Simpler: peak layer must be in top-30% by accuracy.
    above_random = accs - (1/6)
    above_random[above_random < 0] = 0
    if above_random.sum() > 0:
        share_top = above_random[sorted_idx[:top_k]].sum() / above_random.sum()
        gates["tone_concentrated_top30_pct"] = bool(share_top >= 0.50)
        share_top_val = float(share_top)
    else:
        share_top_val = 0.0
        gates["tone_concentrated_top30_pct"] = False

    all_pass = all(v for v in gates.values() if v is not None)
    verdict = "PASS — method spine empirically supported" if all_pass else "FAIL — pivot to backup spine"

    print("\n" + "=" * 70)
    print("GO/NO-GO ANALYSIS")
    print("=" * 70)
    print(f"  Peak layer        : L={peak_layer}  acc={peak_acc:.4f}")
    print(f"  Top {top_k} layers    : {top_layers}  (mean acc={top_mean:.4f})")
    print(f"  Bottom {top_k} layers : {bot_layers}  (mean acc={bot_mean:.4f})")
    print(f"  Best-worst gap    : {best_worst_gap:.4f}   (gate >= {cfg.PASS_BEST_WORST_GAP})")
    print(f"  Tone mass share in top 30%: {share_top_val:.3f}   (gate >= 0.50)")
    print(f"  Random baseline   : {1/6:.4f}")
    print()
    for g, v in gates.items():
        print(f"  [{('PASS' if v else 'FAIL'):>4}]  {g}")
    print()
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    # ----- save -----
    out_dir = Path(cfg.OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.EXP_ID}_whisper_tone_probe.json"
    out_file.write_text(encoding="utf-8", data=json.dumps({
        "config":       asdict(cfg),
        "n_layers":     n_layers,
        "n_utterances": len(utts) - skipped,
        "n_syllables":  int(len(y)),
        "layer_results": layer_results,
        "peak_layer":   peak_layer,
        "peak_acc":     peak_acc,
        "top_layers":   top_layers,
        "top_mean_acc": top_mean,
        "bot_mean_acc": bot_mean,
        "best_worst_gap": best_worst_gap,
        "tone_mass_share_top30": share_top_val,
        "gates":        gates,
        "all_pass":     all_pass,
        "verdict":      verdict,
        "wall_clock_seconds": round(time.perf_counter() - t_start, 1),
    }, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {out_file}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
