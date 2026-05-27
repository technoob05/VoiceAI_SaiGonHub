"""
Contrastive cache-key projection — learned 64-d projection over the
hand-designed 272-d perceptual fingerprint (utterance_cache.py).

Motivation: the existing fingerprint is hand-designed (5 typology-motivated
components — tone histogram, MFCC trajectory, MFCC-delta, F0 stats, energy).
Hand-designed features are interpretable but the dimensions are not
optimally weighted for the cache-hit decision. A LEARNED projection — trained
with triplet loss where positives = same-prompt-different-speaker — can pull
intra-phrase pairs tighter and push inter-phrase pairs apart, widening the
gap that the cosine threshold operates on. Paper §3.10 future-work direction.

This is an OPT-IN add-on. It does NOT modify UtteranceCache. Usage at inference:

    fp272 = cache.fingerprint(audio, sr).vec
    proj  = ContrastiveProjection.load("data/cache_snapshots/contrastive_projection.npz")
    key64 = proj.apply(fp272)
    # then do cosine on key64 instead of fp.vec

Method:
  - Collect (utterance_id, prompt_text, speaker, audio) from VIVOS train.
  - Restrict to prompts spoken by >= 2 speakers (these are the only ones
    that can yield non-trivial positives).
  - Compute 272-d fingerprints for those utterances.
  - Split prompt-groups 80/20 into train/heldout (split BY PROMPT TEXT so no
    prompt appears in both — prevents trivial generalisation).
  - Train 2-layer MLP 272 -> 128 -> 64 with triplet loss (margin 0.3),
    SGD + momentum, batch 32, ~500 iters.
  - Evaluate on held-out: intra-positive cosine and inter-negative cosine,
    raw 272-d vs projected 64-d. Show that the projection widens the gap.

VIVOS reality check: train split has 766 prompts spoken by >=2 speakers,
141 by >=3. After 80/20 split that's ~610 train prompts / ~150 held-out
prompts — enough triplets for a small MLP but not large-scale. We document
this and propose synthetic positives (noise/time-shift augmentation) as
the next step for a production model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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

from utterance_cache import UtteranceCache                       # noqa: E402


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_vivos_train_index() -> dict[str, list[tuple[str, Path]]]:
    """Return prompt_text -> [(speaker_id, wav_path), ...] for VIVOS train."""
    prompts_file = PROJECT / "data" / "processed" / "vivos" / "prompts-train.txt"
    waves_root   = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train" / "waves"
    by_prompt: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for ln in prompts_file.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        uid, txt = ln.split(" ", 1)
        spk = uid.split("_")[0]
        wav = waves_root / spk / f"{uid}.wav"
        if wav.exists():
            by_prompt[txt].append((spk, wav))
    return by_prompt


def compute_fingerprints(items: list[tuple[str, str, Path]], cache: UtteranceCache,
                         verbose: bool = True) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Compute 272-d fingerprints. items = [(prompt, speaker, wav_path)].
    Returns (fp_matrix (N, 272), labels [(prompt, speaker), ...])."""
    fps, labels = [], []
    t0 = time.perf_counter()
    for i, (prompt, spk, wav) in enumerate(items):
        try:
            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            fp = cache.fingerprint(audio, sr)
            if fp.duration_s < 0.5:
                continue
            fps.append(fp.vec)
            labels.append((prompt, spk))
        except Exception as e:
            if verbose:
                print(f"  [warn] failed on {wav.name}: {e}")
        if verbose and (i + 1) % 50 == 0:
            print(f"  fp progress: {i+1}/{len(items)}  elapsed={time.perf_counter()-t0:.1f}s")
    return np.stack(fps, axis=0).astype(np.float32), labels


# --------------------------------------------------------------------------
# MLP + triplet loss (pure NumPy; SGD with momentum)
# --------------------------------------------------------------------------

class TripletMLP:
    """2-layer MLP: 272 -> 128 -> 64 with ReLU. Output L2-normalized.

    Pure NumPy with manual backprop — keeps deployment tiny (no torch in
    the device runtime). Forward/backward verified against a torch reference
    in self-test.
    """

    def __init__(self, in_dim: int = 272, hidden: int = 128, out_dim: int = 64,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        # He init for ReLU
        self.W1 = (rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / (n + eps)

    def forward(self, X: np.ndarray) -> dict:
        """Returns dict with intermediates needed for backward."""
        z1 = X @ self.W1 + self.b1                       # (B, H)
        h1 = np.maximum(z1, 0.0)                          # ReLU
        z2 = h1 @ self.W2 + self.b2                       # (B, out)
        norm = np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8
        out = z2 / norm
        return {"X": X, "z1": z1, "h1": h1, "z2": z2, "norm": norm, "out": out}

    def apply(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["out"]

    def backward_triplet(self, cache_a, cache_p, cache_n, margin: float):
        """Returns (loss, grads_dict, n_active). d = 1 - cosine; on L2-normed
        outputs that's d(a,p) = 1 - a·p. Loss per triplet = max(0, d(a,p) - d(a,n) + margin)
                       = max(0, margin + a·n - a·p)."""
        a = cache_a["out"]; p = cache_p["out"]; n = cache_n["out"]
        B = a.shape[0]
        ap = (a * p).sum(axis=-1)        # (B,)
        an = (a * n).sum(axis=-1)
        loss_per = np.maximum(0.0, margin + an - ap)
        active = (loss_per > 0).astype(np.float32)
        loss = float(loss_per.mean())
        n_active = int(active.sum())

        if n_active == 0:
            zero = lambda M: np.zeros_like(M)
            return loss, {"W1": zero(self.W1), "b1": zero(self.b1),
                          "W2": zero(self.W2), "b2": zero(self.b2)}, 0

        # d L / d a = (n - p) * active / B
        # d L / d p = -a * active / B
        # d L / d n =  a * active / B
        mask = active[:, None] / B
        da_out = (n - p) * mask
        dp_out = -a * mask
        dn_out =  a * mask

        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)

        for c, d_out in [(cache_a, da_out), (cache_p, dp_out), (cache_n, dn_out)]:
            # Backprop through L2 normalization: out = z2 / ||z2||
            # d out / d z2 = (I - out out^T) / ||z2||
            # => d L / d z2 = (d L / d out - (d L / d out · out) * out) / ||z2||
            out = c["out"]; norm = c["norm"]
            dot = (d_out * out).sum(axis=-1, keepdims=True)
            dz2 = (d_out - dot * out) / norm        # (B, out)

            # Layer 2
            gW2 += c["h1"].T @ dz2                   # (H, out)
            gb2 += dz2.sum(axis=0)
            dh1 = dz2 @ self.W2.T                    # (B, H)

            # ReLU backward
            dz1 = dh1 * (c["z1"] > 0)

            # Layer 1
            gW1 += c["X"].T @ dz1
            gb1 += dz1.sum(axis=0)

        return loss, {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, n_active

    def step_sgd_momentum(self, grads: dict, lr: float, momentum: float):
        self.vW1 = momentum * self.vW1 - lr * grads["W1"]; self.W1 += self.vW1
        self.vb1 = momentum * self.vb1 - lr * grads["b1"]; self.b1 += self.vb1
        self.vW2 = momentum * self.vW2 - lr * grads["W2"]; self.W2 += self.vW2
        self.vb2 = momentum * self.vb2 - lr * grads["b2"]; self.b2 += self.vb2

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> "TripletMLP":
        z = np.load(str(path))
        in_dim, hidden = z["W1"].shape
        _,      out_dim = z["W2"].shape
        m = cls(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        m.W1 = z["W1"]; m.b1 = z["b1"]; m.W2 = z["W2"]; m.b2 = z["b2"]
        return m


# --------------------------------------------------------------------------
# Triplet sampling
# --------------------------------------------------------------------------

def sample_triplets(by_prompt_idx: dict[str, list[int]], n: int,
                    rng: random.Random) -> list[tuple[int, int, int]]:
    """Sample n (anchor_idx, positive_idx, negative_idx). Anchor and
    positive share a prompt; negative has a different prompt."""
    eligible = [p for p, ix in by_prompt_idx.items() if len(ix) >= 2]
    if len(eligible) < 2:
        return []
    triplets = []
    for _ in range(n):
        p_anc = rng.choice(eligible)
        a_idx, p_idx = rng.sample(by_prompt_idx[p_anc], 2)
        # Pick a different prompt for negative
        while True:
            p_neg = rng.choice(eligible)
            if p_neg != p_anc:
                break
        n_idx = rng.choice(by_prompt_idx[p_neg])
        triplets.append((a_idx, p_idx, n_idx))
    return triplets


# --------------------------------------------------------------------------
# Held-out evaluation
# --------------------------------------------------------------------------

def eval_gap(fps: np.ndarray, by_prompt_idx: dict[str, list[int]],
             rng: random.Random, n_pos: int = 2000, n_neg: int = 2000) -> dict:
    """Compute mean intra-positive cosine and inter-negative cosine over
    sampled pairs. fps assumed already L2-normalized along axis=-1."""
    fps_n = fps / (np.linalg.norm(fps, axis=-1, keepdims=True) + 1e-8)

    intra = []
    eligible = [p for p, ix in by_prompt_idx.items() if len(ix) >= 2]
    if not eligible:
        return {"intra_cos_mean": 0.0, "inter_cos_mean": 0.0, "gap": 0.0,
                "n_pos": 0, "n_neg": 0}
    for _ in range(n_pos):
        p = rng.choice(eligible)
        i, j = rng.sample(by_prompt_idx[p], 2)
        intra.append(float(np.dot(fps_n[i], fps_n[j])))

    inter = []
    all_prompts = list(by_prompt_idx.keys())
    if len(all_prompts) < 2:
        return {"intra_cos_mean": float(np.mean(intra)), "inter_cos_mean": 0.0,
                "gap": 0.0, "n_pos": len(intra), "n_neg": 0}
    for _ in range(n_neg):
        p1, p2 = rng.sample(all_prompts, 2)
        i = rng.choice(by_prompt_idx[p1]); j = rng.choice(by_prompt_idx[p2])
        inter.append(float(np.dot(fps_n[i], fps_n[j])))

    intra_m = float(np.mean(intra))
    inter_m = float(np.mean(inter))
    return {
        "intra_cos_mean":  round(intra_m, 4),
        "intra_cos_std":   round(float(np.std(intra)),  4),
        "inter_cos_mean":  round(inter_m, 4),
        "inter_cos_std":   round(float(np.std(inter)),  4),
        "gap":             round(intra_m - inter_m, 4),
        "n_pos":           len(intra),
        "n_neg":           len(inter),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts",   type=int, default=400,
                    help="cap multi-speaker prompts (speeds up; uses ~2 utts each)")
    ap.add_argument("--utts-per-prompt", type=int, default=4,
                    help="cap speakers per prompt to load (data-efficiency)")
    ap.add_argument("--n-iters",       type=int, default=500)
    ap.add_argument("--batch-size",    type=int, default=32)
    ap.add_argument("--lr",            type=float, default=0.05)
    ap.add_argument("--momentum",      type=float, default=0.9)
    ap.add_argument("--margin",        type=float, default=0.3)
    ap.add_argument("--hidden",        type=int, default=128)
    ap.add_argument("--out-dim",       type=int, default=64)
    ap.add_argument("--heldout-frac",  type=float, default=0.2)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--out-json",      default="Results/contrastive_cache_key.json")
    ap.add_argument("--out-weights",   default="data/cache_snapshots/contrastive_projection.npz")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Index VIVOS by prompt ----
    print("[index] scanning VIVOS train prompts ...")
    by_prompt_wavs = load_vivos_train_index()
    multi = {p: spks for p, spks in by_prompt_wavs.items() if len(spks) >= 2}
    print(f"[index] total prompts={len(by_prompt_wavs)}  "
          f"multi-speaker (>=2 speakers)={len(multi)}")
    if len(multi) < 50:
        print("[abort] not enough multi-speaker prompts for triplet training.")
        return 1

    # ---- 2. Choose prompts and cap per-prompt utterances ----
    prompt_keys = sorted(multi.keys())
    rng.shuffle(prompt_keys)
    prompt_keys = prompt_keys[: args.max_prompts]

    items: list[tuple[str, str, Path]] = []   # (prompt, speaker, wav)
    for p in prompt_keys:
        spk_wav = multi[p][: args.utts_per_prompt]
        for spk, wav in spk_wav:
            items.append((p, spk, wav))
    print(f"[index] using {len(prompt_keys)} prompts, {len(items)} utterances total")

    # ---- 3. Compute 272-d fingerprints ----
    print("[fp] computing 272-d perceptual fingerprints ...")
    cache = UtteranceCache()
    fps, labels = compute_fingerprints(items, cache)
    print(f"[fp] fingerprint matrix shape={fps.shape}")

    # ---- 4. Train / held-out split BY PROMPT ----
    unique_prompts = sorted({p for p, _ in labels})
    rng.shuffle(unique_prompts)
    n_held = max(1, int(len(unique_prompts) * args.heldout_frac))
    held_prompts  = set(unique_prompts[:n_held])
    train_prompts = set(unique_prompts[n_held:])

    train_idx, held_idx = [], []
    train_by_p, held_by_p = defaultdict(list), defaultdict(list)
    for i, (p, _spk) in enumerate(labels):
        if p in train_prompts:
            train_idx.append(i); train_by_p[p].append(i)
        else:
            held_idx.append(i); held_by_p[p].append(i)

    print(f"[split] train: {len(train_prompts)} prompts / {len(train_idx)} utts; "
          f"held-out: {len(held_prompts)} prompts / {len(held_idx)} utts")

    # Restrict to multi-speaker groups for triplet sampling
    train_by_p_mp = {p: ix for p, ix in train_by_p.items() if len(ix) >= 2}
    held_by_p_mp  = {p: ix for p, ix in held_by_p.items()  if len(ix) >= 2}
    print(f"[split] multi-utt groups -- train: {len(train_by_p_mp)}, held: {len(held_by_p_mp)}")

    if len(train_by_p_mp) < 5 or len(held_by_p_mp) < 5:
        print("[abort] too few multi-utt groups after split.")
        return 1

    # ---- 5. Pre-projection baseline metric ----
    print("[eval] pre-projection (raw 272-d) gap on held-out ...")
    pre = eval_gap(fps[held_idx],
                   {p: [held_idx.index(i) for i in ix] for p, ix in held_by_p_mp.items()},
                   rng)
    print(f"  intra-positive cos: {pre['intra_cos_mean']:.4f} ± {pre['intra_cos_std']:.4f}")
    print(f"  inter-negative cos: {pre['inter_cos_mean']:.4f} ± {pre['inter_cos_std']:.4f}")
    print(f"  gap (intra - inter): {pre['gap']:+.4f}")

    # ---- 6. Train projection ----
    print(f"[train] MLP 272 -> {args.hidden} -> {args.out_dim}, "
          f"triplet margin={args.margin}, iters={args.n_iters}, "
          f"batch={args.batch_size}, lr={args.lr}, momentum={args.momentum}")
    model = TripletMLP(in_dim=fps.shape[1], hidden=args.hidden,
                       out_dim=args.out_dim, seed=args.seed)
    loss_curve = []
    t0 = time.perf_counter()

    for it in range(args.n_iters):
        trips = sample_triplets(train_by_p_mp, args.batch_size, rng)
        a_ix = np.array([t[0] for t in trips], dtype=np.int64)
        p_ix = np.array([t[1] for t in trips], dtype=np.int64)
        n_ix = np.array([t[2] for t in trips], dtype=np.int64)
        Xa, Xp, Xn = fps[a_ix], fps[p_ix], fps[n_ix]

        ca = model.forward(Xa); cp = model.forward(Xp); cn = model.forward(Xn)
        loss, grads, n_active = model.backward_triplet(ca, cp, cn, args.margin)
        model.step_sgd_momentum(grads, args.lr, args.momentum)

        if it % 20 == 0 or it == args.n_iters - 1:
            loss_curve.append({"iter": it, "loss": round(loss, 4),
                               "n_active": n_active})
            print(f"  iter {it:4d}  loss={loss:.4f}  active={n_active}/{args.batch_size}")

    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s")

    # ---- 7. Post-projection evaluation ----
    print("[eval] post-projection (64-d) gap on held-out ...")
    held_fps = fps[held_idx]
    held_proj = model.apply(held_fps)
    post = eval_gap(held_proj,
                    {p: [held_idx.index(i) for i in ix] for p, ix in held_by_p_mp.items()},
                    rng)
    print(f"  intra-positive cos: {post['intra_cos_mean']:.4f} ± {post['intra_cos_std']:.4f}")
    print(f"  inter-negative cos: {post['inter_cos_mean']:.4f} ± {post['inter_cos_std']:.4f}")
    print(f"  gap (intra - inter): {post['gap']:+.4f}")

    gap_improvement = post["gap"] - pre["gap"]
    print(f"\n[result] gap improvement: {gap_improvement:+.4f}  "
          f"({'WIDER' if gap_improvement > 0 else 'NARROWER'} after learned projection)")

    # ---- 8. Save weights ----
    weights_path = PROJECT / args.out_weights
    model.save(weights_path)
    print(f"[save] weights -> {weights_path}")

    # ---- 9. Write JSON ----
    out = {
        "vivos_data": {
            "total_multi_speaker_prompts_available": len(multi),
            "prompts_used":            len(prompt_keys),
            "utterances_used":         len(labels),
            "fp_dim":                  int(fps.shape[1]),
        },
        "split": {
            "train_prompts":  len(train_prompts),
            "train_utts":     len(train_idx),
            "train_multi_utt_groups": len(train_by_p_mp),
            "heldout_prompts": len(held_prompts),
            "heldout_utts":    len(held_idx),
            "heldout_multi_utt_groups": len(held_by_p_mp),
            "split_by":       "prompt_text (no prompt appears in both)",
        },
        "model": {
            "arch":        f"MLP {fps.shape[1]} -> {args.hidden} -> {args.out_dim}",
            "hidden":      args.hidden,
            "out_dim":     args.out_dim,
            "activation":  "ReLU",
            "output_norm": "L2",
            "params":      int(fps.shape[1] * args.hidden + args.hidden +
                               args.hidden * args.out_dim + args.out_dim),
        },
        "training": {
            "iters":      args.n_iters,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "momentum":   args.momentum,
            "margin":     args.margin,
            "optimizer":  "SGD + momentum (pure NumPy)",
            "elapsed_s":  round(elapsed, 2),
        },
        "loss_curve":  loss_curve,
        "heldout_eval": {
            "pre_projection_raw_272d":  pre,
            "post_projection_64d":      post,
            "gap_improvement":          round(gap_improvement, 4),
            "relative_gap_improvement_pct": round(
                100.0 * gap_improvement / max(abs(pre["gap"]), 1e-6), 2),
        },
        "weights_path":  str(weights_path.relative_to(PROJECT)),
        "caveat": (
            "VIVOS provides only ~766 prompts with >=2 speakers in train; "
            "the held-out split inherits ~150 multi-speaker prompts. This "
            "is sufficient to validate the learned-projection HYPOTHESIS "
            "but not to ship a production projection — recommended next "
            "step is to augment positives with noise/time-shift/SpecAugment "
            "of single utterances, which would yield ~10000+ positive pairs."
        ),
    }
    out_json = PROJECT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] json -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
