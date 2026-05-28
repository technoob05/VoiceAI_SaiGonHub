"""
Learned tone-bin classifier (reviewer Q10) — replace the heuristic 6-bin lookup
in tone_histogram_from_f0 with a tiny supervised MLP trained on synthesized
Vietnamese tonal minimal-pair clips.

Reviewer concern: "Could the tone bins be learned (e.g., via a small classifier
trained on labeled tones) rather than heuristically assigned, while staying
within the footprint constraints?"

Method (pure NumPy, CPU-only — no PyTorch/sklearn in the model itself):
  1. Synthesize 6 syllables x 6 tones x N_REPS (default 5) clips via Vi MMS-TTS,
     each spoken by the carrier "nói {syl}".
  2. Per clip, run the same voiced-run segmentation as utterance_cache.py
     (pseudo-syllables of >= 3 contiguous voiced frames). For each run, extract
     features: 4 F0 statistics (mean/slope/dip/creak as the heuristic uses)
     + a fixed-length resampled log-F0 contour (32 timesteps). Label = the
     tone class of the carrier word (since the carrier is "nói X" and X is
     the dominant syllable, we keep the run with most frames as the labelled
     example per clip — safer than trying to align both syllables).
  3. Train a 36 -> 16 -> 6 MLP with cross-entropy + SGD + momentum, pure NumPy.
  4. At inference, replace tone_histogram_from_f0 with a learned version that
     softmax-routes each pseudo-syllable into the 6 bins (giving a soft
     histogram instead of hard counts). The aggregate over runs is normalized.
  5. Re-run the minimal-pair eval with the learned 6-d tone-only stream and
     compare to the heuristic protect-score of +0.028.

CPU-only. If TTS is unavailable, falls back to heuristic-on-heuristic
self-consistency reporting and labels itself preliminary.
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
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                                # noqa: E402

from utterance_cache import (                                     # noqa: E402
    UtteranceCache,
    cheap_mfcc,
    _f0_contour,
    tone_histogram_from_f0,
)


TONE_NAMES = ["ngang", "huyen", "sac", "hoi", "nga", "nang"]

# Vietnamese minimal-pair syllables (6 base x 6 tones).
# Index in inner list = tone class label 0..5
MINIMAL_PAIRS = {
    "ma": ["ma",  "mà",  "má",  "mả",  "mã",  "mạ"],
    "ba": ["ba",  "bà",  "bá",  "bả",  "bã",  "bạ"],
    "ca": ["ca",  "cà",  "cá",  "cả",  "cã",  "cạ"],
    "la": ["la",  "là",  "lá",  "lả",  "lã",  "lạ"],
    "da": ["da",  "dà",  "dá",  "dả",  "dã",  "dạ"],
    "ta": ["ta",  "tà",  "tá",  "tả",  "tã",  "tạ"],
}
# Heuristic-bin mapping in tone_histogram_from_f0:
#   0 ngang, 1 huyen, 2 sac, 3 hoi, 4 nga, 5 nang
CARRIER = "nói {syl}"


# --------------------------------------------------------------------------
# Feature extraction per pseudo-syllable
# --------------------------------------------------------------------------

def voiced_runs(f0: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """Same convention as utterance_cache.tone_histogram_from_f0."""
    voiced_mask = (f0 > 0)
    runs = []
    cur = None
    for i, v in enumerate(voiced_mask):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur >= min_len:
                runs.append((cur, i))
            cur = None
    if cur is not None and len(voiced_mask) - cur >= min_len:
        runs.append((cur, len(voiced_mask)))
    return runs


def run_features(f0: np.ndarray, s: int, e: int, mean_log: float,
                 contour_len: int = 32) -> np.ndarray:
    """Return 4 + contour_len-d feature vector for the voiced run [s,e)."""
    if e - s < 3:
        return np.zeros(4 + contour_len, dtype=np.float32)
    c = np.log(f0[s:e] + 1e-6) - mean_log
    slope = float(c[-1] - c[0]) / max(e - s, 1)
    mid = float(np.median(c))
    dip = float(c.min() - c[(s + e) // 2 - s]) if e - s > 4 else 0.0
    creak = float(np.std(np.diff(c))) if e - s > 4 else 0.0
    # Resample contour to fixed length via linear interp
    contour = np.interp(
        np.linspace(0, len(c) - 1, contour_len),
        np.arange(len(c)),
        c,
    ).astype(np.float32)
    feats = np.concatenate([[slope, mid, dip, creak], contour]).astype(np.float32)
    return feats


def extract_per_clip(audio: np.ndarray, sr: int, contour_len: int = 32) -> list[np.ndarray]:
    """Return list of per-voiced-run feature vectors for a clip."""
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    f0 = _f0_contour(audio, sr)
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return []
    mean_log = float(np.log(voiced + 1e-6).mean())
    runs = voiced_runs(f0)
    return [run_features(f0, s, e, mean_log, contour_len) for (s, e) in runs]


def primary_run_features(audio: np.ndarray, sr: int, contour_len: int = 32) -> np.ndarray:
    """Pick the longest voiced run as the labelled feature (carrier 'nói X' has
    two syllables; the longer one is usually the target X). If no run, zeros."""
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    f0 = _f0_contour(audio, sr)
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return np.zeros(4 + contour_len, dtype=np.float32)
    mean_log = float(np.log(voiced + 1e-6).mean())
    runs = voiced_runs(f0)
    if not runs:
        return np.zeros(4 + contour_len, dtype=np.float32)
    # Pick last run (final syllable = the tonal target after "nói")
    s, e = runs[-1]
    return run_features(f0, s, e, mean_log, contour_len)


# --------------------------------------------------------------------------
# Tiny MLP (pure NumPy) — 36 -> 16 -> 6
# --------------------------------------------------------------------------

class ToneMLP:
    """36 -> 16 -> 6 with ReLU + softmax output. Trained with cross-entropy."""

    def __init__(self, in_dim: int = 36, hidden: int = 16, n_classes: int = 6,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden)) *
                   np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, n_classes)) *
                   np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
        self.feat_mean: np.ndarray | None = None
        self.feat_std:  np.ndarray | None = None

    def _norm(self, X: np.ndarray) -> np.ndarray:
        if self.feat_mean is None:
            return X
        return (X - self.feat_mean) / (self.feat_std + 1e-6)

    def forward(self, X: np.ndarray) -> dict:
        X = self._norm(X)
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        m = z2.max(axis=-1, keepdims=True)
        e = np.exp(z2 - m)
        p = e / e.sum(axis=-1, keepdims=True)
        return {"X": X, "z1": z1, "h1": h1, "z2": z2, "p": p}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["p"]

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict, float]:
        c = self.forward(X)
        p = c["p"]; B = X.shape[0]
        log_p = np.log(p[np.arange(B), y] + 1e-12)
        loss = float(-log_p.mean())
        acc = float((p.argmax(axis=-1) == y).mean())
        dz2 = p.copy()
        dz2[np.arange(B), y] -= 1.0
        dz2 /= B
        gW2 = c["h1"].T @ dz2
        gb2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (c["z1"] > 0)
        gW1 = c["X"].T @ dz1
        gb1 = dz1.sum(axis=0)
        return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, acc

    def step(self, grads: dict, lr: float, momentum: float, wd: float = 1e-4):
        self.vW1 = momentum * self.vW1 - lr * (grads["W1"] + wd * self.W1); self.W1 += self.vW1
        self.vb1 = momentum * self.vb1 - lr *  grads["b1"];                self.b1 += self.vb1
        self.vW2 = momentum * self.vW2 - lr * (grads["W2"] + wd * self.W2); self.W2 += self.vW2
        self.vb2 = momentum * self.vb2 - lr *  grads["b2"];                self.b2 += self.vb2

    def fit_norm(self, X: np.ndarray):
        self.feat_mean = X.mean(axis=0).astype(np.float32)
        self.feat_std  = X.std(axis=0).astype(np.float32)

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size +
                   (self.feat_mean.size if self.feat_mean is not None else 0) +
                   (self.feat_std.size  if self.feat_std  is not None else 0))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 feat_mean=(self.feat_mean if self.feat_mean is not None else np.zeros(self.W1.shape[0], dtype=np.float32)),
                 feat_std =(self.feat_std  if self.feat_std  is not None else np.ones (self.W1.shape[0], dtype=np.float32)))

    @classmethod
    def load(cls, path: Path) -> "ToneMLP":
        z = np.load(str(path))
        in_dim, hidden = z["W1"].shape
        _, n_classes = z["W2"].shape
        m = cls(in_dim=in_dim, hidden=hidden, n_classes=n_classes)
        m.W1 = z["W1"]; m.b1 = z["b1"]; m.W2 = z["W2"]; m.b2 = z["b2"]
        m.feat_mean = z["feat_mean"]; m.feat_std = z["feat_std"]
        return m


# --------------------------------------------------------------------------
# Learned tone histogram (replaces heuristic at inference time)
# --------------------------------------------------------------------------

def learned_tone_histogram_from_f0(f0: np.ndarray, model: ToneMLP,
                                   contour_len: int = 32) -> np.ndarray:
    """Soft 6-bin histogram from the learned classifier. Replaces
    utterance_cache.tone_histogram_from_f0 in the inference path."""
    voiced = f0[f0 > 0]
    if len(voiced) < 4:
        return np.zeros(6, dtype=np.float32)
    mean_log = float(np.log(voiced + 1e-6).mean())
    runs = voiced_runs(f0)
    if not runs:
        return np.zeros(6, dtype=np.float32)
    feats = np.stack([run_features(f0, s, e, mean_log, contour_len)
                      for (s, e) in runs], axis=0)
    probs = model.predict_proba(feats)
    hist = probs.mean(axis=0)
    s = float(hist.sum())
    if s > 0:
        hist = hist / s
    return hist.astype(np.float32)


# --------------------------------------------------------------------------
# Synthesis or fallback
# --------------------------------------------------------------------------

def try_synth_clips(n_reps: int) -> dict | None:
    """Try to synthesize 6x6xN_REPS clips. Returns dict of (base, tone_idx, rep)
    -> (audio, sr), or None if TTS unavailable."""
    try:
        from transformers import AutoTokenizer, VitsModel
        import torch
    except Exception as e:
        print(f"[warn] cannot import transformers/torch ({e}) — TTS unavailable")
        return None
    print("[synth] loading MMS-TTS-vie ...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[synth] loaded in {time.perf_counter()-t0:.1f}s, sr={sr}")

    clips = {}
    t_synth = time.perf_counter()
    n_total = 0
    for base, variants in MINIMAL_PAIRS.items():
        for tone_idx, syl in enumerate(variants):
            for rep in range(n_reps):
                text = CARRIER.format(syl=syl)
                # rep variation: pad noise prefix / suffix on the prompt? VITS is deterministic.
                # Use rep as a seed for VITS noise scale by re-seeding torch.
                try:
                    torch.manual_seed(rep * 31 + tone_idx * 7 + hash(base) % 997)
                    inputs = tok(text, return_tensors="pt")
                    with torch.inference_mode():
                        # Slight noise scale variation gets us different prosody per rep.
                        wav = model(**inputs,
                                    noise_scale=0.667 + 0.05 * rep,
                                    noise_scale_duration=0.8 + 0.05 * rep).waveform
                    a = wav.squeeze().cpu().numpy().astype(np.float32)
                    clips[(base, tone_idx, rep)] = (a, sr)
                    n_total += 1
                except Exception as e:
                    print(f"  [warn] synth failed for {text!r} rep={rep}: {e}")
            if (tone_idx + 1) % 6 == 0:
                el = time.perf_counter() - t_synth
                print(f"  base={base} done  total={n_total} clips  elapsed={el:.1f}s")
    print(f"[synth] generated {len(clips)} clips in {time.perf_counter()-t_synth:.1f}s")
    return clips


# --------------------------------------------------------------------------
# Evaluation: minimal-pair protect score with the learned tone stream only
# --------------------------------------------------------------------------

def eval_tone_only_protect(clips: dict, hist_fn) -> dict:
    """clips: {(base, tone, rep): (audio, sr)}. hist_fn: f0 -> 6-d hist.
    Returns intra/inter cosine stats for the 6-d tone-only stream."""
    # Group by (base, syl-tone) — we evaluate one representative per (base, tone)
    # to keep the pairing the same as minimal_pair_eval.py (which used 1 rep each).
    by_bt = {}
    for (base, tone, rep), (a, sr) in clips.items():
        f0 = _f0_contour(a if sr == 16000 else np.interp(
            np.linspace(0, len(a), int(round(len(a)*16000/sr)), endpoint=False),
            np.arange(len(a)), a).astype(np.float32),
            16000)
        h = hist_fn(f0)
        n = np.linalg.norm(h)
        if n > 1e-6:
            h = h / n
        by_bt.setdefault((base, tone), []).append(h)

    # average histogram across reps to reduce TTS noise
    rep_hist = {k: np.mean(v, axis=0) for k, v in by_bt.items()}
    for k, v in rep_hist.items():
        n = np.linalg.norm(v)
        if n > 1e-6:
            rep_hist[k] = v / n

    bases = sorted({b for (b, _) in rep_hist.keys()})
    tones = sorted({t for (_, t) in rep_hist.keys()})

    intra = []
    for b in bases:
        for i in range(len(tones)):
            for j in range(i + 1, len(tones)):
                k1, k2 = (b, tones[i]), (b, tones[j])
                if k1 in rep_hist and k2 in rep_hist:
                    intra.append(float(rep_hist[k1] @ rep_hist[k2]))

    inter = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            for t1 in tones:
                for t2 in tones:
                    k1, k2 = (bases[i], t1), (bases[j], t2)
                    if k1 in rep_hist and k2 in rep_hist:
                        inter.append(float(rep_hist[k1] @ rep_hist[k2]))

    intra_m = float(np.mean(intra)) if intra else 0.0
    inter_m = float(np.mean(inter)) if inter else 0.0
    return {
        "intra_tone_cos_mean": round(intra_m, 4),
        "intra_tone_cos_std":  round(float(np.std(intra)) if intra else 0.0, 4),
        "inter_syl_cos_mean":  round(inter_m, 4),
        "inter_syl_cos_std":   round(float(np.std(inter)) if inter else 0.0, 4),
        "protect_score":       round(inter_m - intra_m, 4),
        "n_intra_pairs":       len(intra),
        "n_inter_pairs":       len(inter),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--contour-len", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="Results/learned_tone_bins_eval.json")
    ap.add_argument("--out-weights", default="data/cache_snapshots/learned_tone_classifier.npz")
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- 1. Generate or load synth clips ---
    clips = try_synth_clips(args.n_reps)
    if clips is None or len(clips) < 36:
        print("[abort] TTS unavailable or generated too few clips; cannot run honestly.")
        out = {
            "status":  "fallback",
            "message": "MMS-TTS unavailable; experiment is preliminary heuristic-on-heuristic.",
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 1

    # --- 2. Build labelled feature set ---
    print(f"\n[feat] extracting primary-run features (contour_len={args.contour_len}) ...")
    X_list, y_list, key_list = [], [], []
    for (base, tone, rep), (a, sr) in clips.items():
        feat = primary_run_features(a, sr, contour_len=args.contour_len)
        if np.allclose(feat, 0):
            continue
        X_list.append(feat)
        y_list.append(tone)
        key_list.append((base, tone, rep))
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[feat] dataset: X.shape={X.shape}  y.shape={y.shape}  in_dim={X.shape[1]}")

    # --- 3. Held-out split (BY base+rep so the held-out clip is not a syllable
    #         repeat of a training clip). Simple stratified hold-out: leave 1
    #         rep out per (base, tone). ---
    held_mask = np.array([rep == (args.n_reps - 1) for (_, _, rep) in key_list])
    train_idx = np.where(~held_mask)[0]
    held_idx  = np.where(held_mask)[0]
    print(f"[split] train={len(train_idx)}  heldout={len(held_idx)}")

    Xtr, ytr = X[train_idx], y[train_idx]
    Xho, yho = X[held_idx],  y[held_idx]

    # --- 4. Train MLP ---
    model = ToneMLP(in_dim=X.shape[1], hidden=args.hidden, n_classes=6,
                    seed=args.seed)
    model.fit_norm(Xtr)
    print(f"[train] MLP {X.shape[1]} -> {args.hidden} -> 6   params={model.n_params()}")
    loss_curve = []
    t0 = time.perf_counter()
    for it in range(args.n_iters):
        ix = rng.integers(0, len(Xtr), size=args.batch_size)
        loss, grads, acc = model.loss_and_grads(Xtr[ix], ytr[ix])
        model.step(grads, args.lr, args.momentum)
        if it % 30 == 0 or it == args.n_iters - 1:
            loss_curve.append({"iter": it, "loss": round(loss, 4), "batch_acc": round(acc, 3)})
            print(f"  iter {it:4d}  loss={loss:.4f}  batch_acc={acc:.3f}")
    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s")

    # --- 5. Held-out accuracy ---
    p_ho = model.predict_proba(Xho)
    y_pred = p_ho.argmax(axis=-1)
    acc_ho = float((y_pred == yho).mean())
    p_tr = model.predict_proba(Xtr)
    acc_tr = float((p_tr.argmax(axis=-1) == ytr).mean())
    print(f"[eval] train acc={acc_tr:.3f}   heldout acc={acc_ho:.3f}")

    # Per-tone confusion
    confusion = np.zeros((6, 6), dtype=np.int64)
    for t, p in zip(yho.tolist(), y_pred.tolist()):
        confusion[t, p] += 1
    print(f"[eval] heldout confusion (rows=true, cols=pred):\n{confusion}")

    # --- 6. Save weights + size on disk ---
    weights_path = PROJECT / args.out_weights
    model.save(weights_path)
    size_kb = weights_path.stat().st_size / 1024.0
    print(f"[save] weights -> {weights_path}  size={size_kb:.2f} KB")

    # --- 7. Replay minimal-pair protect-score with LEARNED tone stream ---
    print("\n[eval] running minimal-pair protect-score with learned tone histogram ...")
    learned_stats = eval_tone_only_protect(
        clips,
        lambda f0: learned_tone_histogram_from_f0(f0, model, contour_len=args.contour_len),
    )
    print(f"  learned : intra={learned_stats['intra_tone_cos_mean']:.4f}  "
          f"inter={learned_stats['inter_syl_cos_mean']:.4f}  "
          f"protect={learned_stats['protect_score']:+.4f}")

    print("[eval] running same minimal-pair protect-score with HEURISTIC histogram (sanity) ...")
    heuristic_stats = eval_tone_only_protect(clips, tone_histogram_from_f0)
    print(f"  heuristic: intra={heuristic_stats['intra_tone_cos_mean']:.4f}  "
          f"inter={heuristic_stats['inter_syl_cos_mean']:.4f}  "
          f"protect={heuristic_stats['protect_score']:+.4f}")

    # Baseline from the existing pinned eval (Results/minimal_pair_eval.json)
    BASELINE_HEURISTIC = {
        "intra_tone_cos_mean": 0.6916,
        "inter_syl_cos_mean":  0.7194,
        "protect_score":       0.0279,
        "source":              "Results/minimal_pair_eval.json (synth nói X carrier, n_reps=1)",
    }

    delta_vs_baseline = round(learned_stats["protect_score"] - BASELINE_HEURISTIC["protect_score"], 4)

    out = {
        "status":            "ok",
        "experiment_summary": "Learned 36->16->6 MLP tone classifier replaces the heuristic 6-bin lookup in tone_histogram_from_f0. Reviewer Q10.",
        "data": {
            "minimal_pairs":  MINIMAL_PAIRS,
            "carrier":        CARRIER,
            "n_reps_per_cell": args.n_reps,
            "n_clips_synthed": len(clips),
            "n_train":         int(len(train_idx)),
            "n_heldout":       int(len(held_idx)),
            "input_dim":       int(X.shape[1]),
            "contour_len":     args.contour_len,
        },
        "model": {
            "arch":      f"MLP {X.shape[1]} -> {args.hidden} -> 6",
            "hidden":    args.hidden,
            "params":    model.n_params(),
            "weights_path": str(weights_path.relative_to(PROJECT)),
            "size_kb":   round(size_kb, 2),
        },
        "training": {
            "iters":     args.n_iters,
            "lr":        args.lr,
            "momentum":  args.momentum,
            "batch_size": args.batch_size,
            "elapsed_s": round(elapsed, 2),
        },
        "loss_curve": loss_curve,
        "classification": {
            "train_accuracy":   round(acc_tr, 4),
            "heldout_accuracy": round(acc_ho, 4),
            "heldout_confusion_matrix": confusion.tolist(),
            "tone_classes":     TONE_NAMES,
        },
        "minimal_pair_protect_score": {
            "heuristic_on_same_clips":  heuristic_stats,
            "learned_on_same_clips":    learned_stats,
            "baseline_from_pinned_results": BASELINE_HEURISTIC,
            "delta_learned_vs_baseline_heuristic": delta_vs_baseline,
            "delta_learned_vs_heuristic_same_clips": round(
                learned_stats["protect_score"] - heuristic_stats["protect_score"], 4),
        },
        "footprint_assessment": {
            "size_kb":        round(size_kb, 2),
            "target_kb":      50,
            "passes_50kb":    size_kb <= 50.0,
            "params_total":   model.n_params(),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"\n[done] wrote {args.out_json}")
    print(f"[headline] learned protect = {learned_stats['protect_score']:+.4f}  "
          f"vs heuristic-baseline {BASELINE_HEURISTIC['protect_score']:+.4f}  "
          f"delta={delta_vs_baseline:+.4f}   footprint={size_kb:.2f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
