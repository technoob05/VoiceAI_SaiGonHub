"""
PhoWhisper tone-layer probe (paper Appendix F).

Extends de la Fuente & Jurafsky (Interspeech 2024) from self-supervised Mandarin
SSL models to PhoWhisper-small, a supervised Vietnamese-fine-tuned Whisper. The
output answers: which encoder layer of PhoWhisper-small carries the most
Vietnamese tone-classification signal, and is that layer a plausible attach-point
for a learned tone stream in the two-stream voting cache (Paper §3.5)?

Protocol:
  1. Synthesize 12 syllables x 6 tones x 5 reps = 360 clips via MMS-TTS-vie,
     carrier phrase "noi {syl}", deterministic VITS noise-scale per rep (seed=42).
  2. For each clip, log-mel feature -> PhoWhisper-small encoder forward with
     output_hidden_states=True. Twelve hidden_states[1..12] (post-layer
     activations of the 12 encoder layers); mean-pool over time -> 768-d
     per (clip, layer).
  3. Per layer, train a logistic regression (768->6 softmax, pure NumPy SGD)
     with 80/20 stratified train/test split. Report 6-way tone accuracy.
  4. Plot accuracy vs layer index, save to papers/figures/.
  5. Save per-layer accuracies, best-layer confusion matrix, and pointer to
     where a learned tone stream should attach.

CPU-only. ~244MB model + 360 forward passes ~ 6-15 minutes wall time.

Fallback chain:
  HF PhoWhisper-small -> wav2vec2-base. (faster-whisper does not expose
  per-layer encoder hidden states cleanly, so we go HF -> w2v2 directly.)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
(HF_CACHE / "xet").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME",            str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",       str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE",       str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE",     "0")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

_LOG_PATH = Path(__file__).resolve().parents[1] / "Results" / "phowhisper_tone_probe.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOG_FH = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass
    def isatty(self):
        for st in self.streams:
            try:
                if st.isatty(): return True
            except Exception: pass
        return False
    def fileno(self):
        for st in self.streams:
            try: return st.fileno()
            except Exception: continue
        raise OSError("no fileno")
    @property
    def encoding(self):
        for st in self.streams:
            enc = getattr(st, "encoding", None)
            if enc: return enc
        return "utf-8"
    def writable(self): return True
    def readable(self): return False
    def close(self):
        for st in self.streams:
            try: st.close()
            except Exception: pass

sys.stdout = _Tee(sys.__stdout__, _LOG_FH)
sys.stderr = _Tee(sys.__stderr__, _LOG_FH)

import numpy as np  # noqa: E402

TONE_NAMES = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]

# 12 carrier-syllable bases x 6 tones. Each tone index 0..5 maps to:
#   0 ngang, 1 huyen, 2 sac, 3 hoi, 4 nga, 5 nang.
MINIMAL_PAIRS = {
    "ma": ["ma",  "mà",  "má",  "mả",  "mã",  "mạ"],
    "ba": ["ba",  "bà",  "bá",  "bả",  "bã",  "bạ"],
    "ca": ["ca",  "cà",  "cá",  "cả",  "cã",  "cạ"],
    "la": ["la",  "là",  "lá",  "lả",  "lã",  "lạ"],
    "da": ["da",  "dà",  "dá",  "dả",  "dã",  "dạ"],
    "ta": ["ta",  "tà",  "tá",  "tả",  "tã",  "tạ"],
    "na": ["na",  "nà",  "ná",  "nả",  "nã",  "nạ"],
    "va": ["va",  "và",  "vá",  "vả",  "vã",  "vạ"],
    "sa": ["sa",  "sà",  "sá",  "sả",  "sã",  "sạ"],
    "ha": ["ha",  "hà",  "há",  "hả",  "hã",  "hạ"],
    "ga": ["ga",  "gà",  "gá",  "gả",  "gã",  "gạ"],
    "ra": ["ra",  "rà",  "rá",  "rả",  "rã",  "rạ"],
}
CARRIER = "nói {syl}"
N_REPS = 3
SEED = 42


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

def synthesize_clips(n_reps: int = N_REPS):
    """Yield (base, tone_idx, rep, audio_np_float32, sr). Deterministic."""
    from transformers import AutoTokenizer, VitsModel
    import torch
    print("[synth] loading MMS-TTS-vie ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[synth] loaded in {time.perf_counter()-t0:.1f}s, sr={sr}", flush=True)

    clips = []
    t_synth = time.perf_counter()
    for base, variants in MINIMAL_PAIRS.items():
        for tone_idx, syl in enumerate(variants):
            for rep in range(n_reps):
                text = CARRIER.format(syl=syl)
                ids = tok(text, return_tensors="pt").input_ids
                # Per-rep noise-scale variation (deterministic) - perturb VITS prior
                torch.manual_seed(SEED + 1000 * tone_idx + 17 * rep + hash(base) % 7919)
                with torch.no_grad():
                    out = model(ids).waveform
                audio = out.squeeze().cpu().numpy().astype(np.float32)
                clips.append((base, tone_idx, rep, audio, sr))
    print(f"[synth] {len(clips)} clips in {time.perf_counter()-t_synth:.1f}s",
          flush=True)
    return clips


# --------------------------------------------------------------------------
# Per-layer activations
# --------------------------------------------------------------------------

def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    n = int(round(len(audio) * 16000.0 / sr))
    out = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                    np.arange(len(audio)), audio).astype(np.float32)
    return out


def extract_phowhisper_activations(clips, model_name="vinai/PhoWhisper-small"):
    """Return X: dict[layer_idx -> (N, D)], y: (N,), labels."""
    import torch
    from transformers import WhisperModel, WhisperFeatureExtractor
    print(f"[probe] loading {model_name} ...", flush=True)
    t0 = time.perf_counter()
    feat = WhisperFeatureExtractor.from_pretrained(model_name)
    mdl  = WhisperModel.from_pretrained(model_name).eval()
    # Number of encoder layers
    n_layers = mdl.config.encoder_layers
    d_model  = mdl.config.d_model
    print(f"[probe] loaded in {time.perf_counter()-t0:.1f}s "
          f"layers={n_layers} d_model={d_model}", flush=True)

    # Storage: hidden_states[0] is the input embedding; hidden_states[1..n_layers]
    # are post-each-encoder-layer activations. We probe layers 1..n_layers.
    per_layer = [[] for _ in range(n_layers)]
    labels = []
    bases  = []

    t_enc = time.perf_counter()
    with torch.no_grad():
        for i, (base, tone_idx, rep, audio, sr) in enumerate(clips):
            x = resample_to_16k(audio, sr)
            inp = feat(x, sampling_rate=16000, return_tensors="pt")
            # WhisperModel.encoder forward
            out = mdl.encoder(
                inp.input_features,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states  # tuple len n_layers+1
            for L in range(n_layers):
                # hidden_states[L+1] = post-layer-L activations; shape (1, T, D)
                pooled = hs[L+1].squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
                per_layer[L].append(pooled)
            labels.append(tone_idx)
            bases.append(base)
            if (i+1) % 30 == 0:
                dt = time.perf_counter() - t_enc
                eta = dt * (len(clips) - i - 1) / (i + 1)
                print(f"[probe] {i+1}/{len(clips)} clips, "
                      f"elapsed={dt:.0f}s eta={eta:.0f}s", flush=True)
    X = {L: np.stack(per_layer[L], axis=0) for L in range(n_layers)}
    y = np.array(labels, dtype=np.int64)
    return X, y, np.array(bases), n_layers, d_model


# --------------------------------------------------------------------------
# Tiny linear probe (768 -> 6 softmax) trained in pure NumPy
# --------------------------------------------------------------------------

def train_linear_probe(X_train, y_train, X_test, y_test,
                       n_iter=2000, lr=0.1, wd=1e-3, seed=0):
    """Multinomial logistic regression with L2, SGD with full-batch on small N."""
    rng = np.random.default_rng(seed)
    D = X_train.shape[1]
    C = int(y_train.max()) + 1
    # Standardize
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0) + 1e-6
    Xtr = (X_train - mu) / sd
    Xte = (X_test  - mu) / sd
    W = (rng.standard_normal((D, C)) * np.sqrt(1.0 / D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    Y = np.eye(C, dtype=np.float32)[y_train]
    N = Xtr.shape[0]
    for it in range(n_iter):
        logits = Xtr @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits); P = e / e.sum(axis=1, keepdims=True)
        # gradient
        gW = Xtr.T @ (P - Y) / N + wd * W
        gb = (P - Y).mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    logits_te = Xte @ W + b
    pred_te = logits_te.argmax(axis=1)
    logits_tr = Xtr @ W + b
    pred_tr = logits_tr.argmax(axis=1)
    acc_te = float((pred_te == y_test).mean())
    acc_tr = float((pred_tr == y_train).mean())
    # Confusion matrix on test
    C = int(max(y_test.max(), y_train.max())) + 1
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_test, pred_te):
        cm[t, p] += 1
    n_params = int(W.size + b.size)
    return acc_tr, acc_te, cm, n_params


def stratified_split(y: np.ndarray, frac_train: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx_tr, idx_te = [], []
    for c in np.unique(y):
        ii = np.where(y == c)[0]
        rng.shuffle(ii)
        k = int(round(len(ii) * frac_train))
        idx_tr.extend(ii[:k].tolist())
        idx_te.extend(ii[k:].tolist())
    return np.array(idx_tr, dtype=np.int64), np.array(idx_te, dtype=np.int64)


# --------------------------------------------------------------------------
# Fallback: wav2vec2-base (12 layers, 768-d)
# --------------------------------------------------------------------------

def extract_w2v2_activations(clips, model_name="facebook/wav2vec2-base"):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    print(f"[fallback] loading {model_name} ...", flush=True)
    feat = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    mdl  = Wav2Vec2Model.from_pretrained(model_name).eval()
    n_layers = mdl.config.num_hidden_layers
    d_model  = mdl.config.hidden_size
    print(f"[fallback] layers={n_layers} d={d_model}", flush=True)
    per_layer = [[] for _ in range(n_layers)]
    labels = []
    t_enc = time.perf_counter()
    with torch.no_grad():
        for i, (base, tone_idx, rep, audio, sr) in enumerate(clips):
            x = resample_to_16k(audio, sr)
            inp = feat(x, sampling_rate=16000, return_tensors="pt")
            out = mdl(inp.input_values, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            for L in range(n_layers):
                pooled = hs[L+1].squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
                per_layer[L].append(pooled)
            labels.append(tone_idx)
            if (i+1) % 30 == 0:
                dt = time.perf_counter() - t_enc
                print(f"[fallback] {i+1}/{len(clips)} elapsed={dt:.0f}s", flush=True)
    X = {L: np.stack(per_layer[L], axis=0) for L in range(n_layers)}
    y = np.array(labels, dtype=np.int64)
    return X, y, n_layers, d_model


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_layerwise(layer_acc: list, fig_path: Path, model_label: str,
                   best_layer: int, peak_acc: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    layers = np.arange(1, len(layer_acc) + 1)
    ax.plot(layers, np.array(layer_acc) * 100, marker="o", linewidth=2,
            color="#2b6cb0", label=f"{model_label} (held-out)")
    ax.axhline(100/6, linestyle="--", color="gray", linewidth=1,
               label=f"random ({100/6:.2f}%)")
    ax.axvline(best_layer, linestyle=":", color="#c05621", linewidth=1.4,
               label=f"peak L{best_layer} = {peak_acc*100:.1f}%")
    # de la Fuente & Jurafsky layer-8 reference on Mandarin SSL models
    ax.axvline(8, linestyle=":", color="#2f855a", linewidth=1,
               alpha=0.7, label="de la Fuente '24 L8 (Mandarin SSL)")
    ax.set_xlabel("Encoder layer index")
    ax.set_ylabel("6-way tone accuracy (%)")
    ax.set_title(f"Vietnamese tone-classification accuracy per encoder layer "
                 f"({model_label})")
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    print(f"[plot] saved {fig_path}", flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    out_json = PROJECT / "Results" / "phowhisper_tone_probe.json"
    fig_path = PROJECT / "papers" / "figures" / "fig_phowhisper_tone_probe.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    clips = synthesize_clips(N_REPS)

    model_used = "vinai/PhoWhisper-small"
    fallback_used = None
    try:
        X, y, bases, n_layers, d_model = extract_phowhisper_activations(
            clips, model_used)
    except Exception as e:
        import traceback
        print(f"[warn] PhoWhisper load failed: {e}", flush=True)
        traceback.print_exc()
        try:
            print("[fallback] trying openai/whisper-small ...", flush=True)
            X, y, bases, n_layers, d_model = extract_phowhisper_activations(
                clips, "openai/whisper-small")
            model_used = "openai/whisper-small"
            fallback_used = "whisper-small"
        except Exception as e2:
            print(f"[warn] openai/whisper-small failed: {e2}", flush=True)
            traceback.print_exc()
            try:
                print("[fallback] trying facebook/wav2vec2-base ...", flush=True)
                X, y, n_layers, d_model = extract_w2v2_activations(
                    clips, "facebook/wav2vec2-base")
                bases = np.array([c[0] for c in clips])
                model_used = "facebook/wav2vec2-base"
                fallback_used = "wav2vec2-base"
            except Exception as e3:
                print(f"[error] all fallbacks failed: {e3}", flush=True)
                traceback.print_exc()
                raise

    print(f"[probe] activations done; running per-layer probes ...", flush=True)
    idx_tr, idx_te = stratified_split(y, frac_train=0.8, seed=SEED)
    layer_results = []
    best_layer, best_acc, best_cm, best_params = -1, -1.0, None, 0
    for L in range(n_layers):
        Xtr = X[L][idx_tr]; ytr = y[idx_tr]
        Xte = X[L][idx_te]; yte = y[idx_te]
        acc_tr, acc_te, cm, n_params = train_linear_probe(
            Xtr, ytr, Xte, yte, n_iter=1500, lr=0.1, wd=1e-3, seed=L)
        layer_results.append({
            "layer": L + 1,
            "train_acc": acc_tr,
            "test_acc": acc_te,
            "n_params": n_params,
        })
        print(f"  L{L+1:2d}  train={acc_tr*100:5.1f}%  test={acc_te*100:5.1f}%",
              flush=True)
        if acc_te > best_acc:
            best_acc = acc_te; best_layer = L + 1; best_cm = cm.tolist()
            best_params = n_params

    # Plot
    plot_layerwise([r["test_acc"] for r in layer_results], fig_path,
                   model_used.split("/")[-1], best_layer, best_acc)

    # Model footprint
    model_blob = HF_CACHE / "hub" / f"models--{model_used.replace('/', '--')}"
    model_size_mb = 0.0
    if model_blob.exists():
        for p in model_blob.rglob("*"):
            if p.is_file():
                try:
                    model_size_mb += p.stat().st_size
                except OSError:
                    pass
        model_size_mb = model_size_mb / (1024 * 1024)

    # Probe weights footprint per layer (W: 768x6 + b: 6 floats float32)
    probe_kb = (d_model * 6 + 6) * 4 / 1024.0

    # de la Fuente reference: layer 8 peak on wav2vec2/HuBERT/WavLM for Mandarin
    delafuente_ref_layer = 8

    record = {
        "model": model_used,
        "fallback_used": fallback_used,
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "n_clips": int(len(clips)),
        "syllables": list(MINIMAL_PAIRS.keys()),
        "n_train": int(len(idx_tr)),
        "n_test":  int(len(idx_te)),
        "random_baseline": 1.0 / 6,
        "per_layer": layer_results,
        "best_layer": int(best_layer),
        "best_test_acc": float(best_acc),
        "best_layer_confusion_matrix": best_cm,
        "tone_names": TONE_NAMES,
        "delafuente_jurafsky_2024_reference_layer": delafuente_ref_layer,
        "model_footprint_mb": float(model_size_mb),
        "probe_weights_per_layer_kb": float(probe_kb),
        "wall_time_s": float(time.perf_counter() - t_start),
        "seed": SEED,
        "n_reps_per_cell": N_REPS,
        "figure_path": str(fig_path.relative_to(PROJECT)).replace("\\", "/"),
    }
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[done] saved {out_json}", flush=True)
    print(f"[done] best layer = L{best_layer}, test acc = {best_acc*100:.2f}% "
          f"vs random {100/6:.2f}%, vs de la Fuente Mandarin L8", flush=True)
    print(f"[done] wall = {record['wall_time_s']:.0f}s", flush=True)

    # ---- Verify outputs exist with non-zero size ----
    print("\n[verify] checking output artifacts", flush=True)
    ok = True
    for path in (out_json, fig_path):
        abs_p = path.resolve()
        if not abs_p.exists():
            print(f"[verify] MISSING: {abs_p}", flush=True)
            ok = False
            continue
        sz = abs_p.stat().st_size
        if sz <= 0:
            print(f"[verify] ZERO-BYTE: {abs_p}", flush=True)
            ok = False
            continue
        print(f"[verify] OK  {abs_p}  ({sz} bytes)", flush=True)
    if not ok:
        raise SystemExit("[verify] one or more outputs missing or empty")
    print("[verify] all outputs present and non-empty", flush=True)
    return record


if __name__ == "__main__":
    main()
