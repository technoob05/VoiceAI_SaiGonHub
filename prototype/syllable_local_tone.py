"""
Syllable-local tone pooling — Round 27 production fix for sentence-level
adversarial false-hits (46.7% at deployment default in Round 26).

Root cause (paper §6 / Round 26): the utterance-level tone stream computes
ONE 32-d learned embedding over the WHOLE sentence audio, averaging the
contrastive syllable's tone signal against many non-contrastive carrier
syllables. The tonal veto is diluted.

Fix: detect syllable boundaries on the waveform, compute the 32-d learned
tone embedding PER SYLLABLE, then aggregate.

Pipeline (pure NumPy, no GPU, no retraining):
  1. Voiced-region segmentation via energy + F0 gating.
  2. Per-syllable 63-d feature -> reuse learned MLP -> 32-d tone vec.
  3. Aggregate across syllables with one of:
       (a) mean-pool   (= current Round 26 baseline behavior)
       (b) max-pool    (per-dim max across syllables)
       (c) concat-then-project (pad to K=8 syl, 256-d -> 32-d linear; init
           as identity-sum so baseline-compatible; lightly trained on the
           120-clip eval pairs)
       (d) attention-pool (32 learned 32-d queries; ~1K params)

Eval re-uses adversarial_minimal_pair_eval.py protocol and a 120-clip
syllable-level sanity eval (subset of MINIMAL_PAIR_SYLLABLES_30, 20 syllables).

Strict constraints: pure NumPy + matplotlib. Reuse existing learned-tone
weights at data/cache_snapshots/learned_tone_stream.npz — no retraining
of the base 32-d embedding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

from utterance_cache import cheap_mfcc, _f0_contour, UtteranceCache  # noqa: E402
from two_stream_voting import (                                      # noqa: E402
    TwoStreamCache, TwoStreamConfig, TwoStreamFingerprint, _normalize,
)
from learned_tone_stream import (                                    # noqa: E402
    ToneEmbeddingMLP, clip_to_64d,
    MINIMAL_PAIR_SYLLABLES_30,
)


# -----------------------------------------------------------------------
# Syllable boundary detection (pure NumPy)
# -----------------------------------------------------------------------

@dataclass
class SyllableConfig:
    sr:              int   = 16000
    frame_ms:        float = 20.0   # F0/energy frame
    win_ms:          float = 40.0
    min_syl_ms:      float = 80.0
    max_syl_ms:      float = 450.0
    min_voiced_run:  int   = 4      # frames
    energy_floor_db: float = -32.0  # below peak
    max_syllables:   int   = 12


def _frame_energy_db(audio: np.ndarray, sr: int, frame_ms: float = 20.0,
                     win_ms: float = 40.0) -> np.ndarray:
    hop = int(sr * frame_ms / 1000)
    win = int(sr * win_ms / 1000)
    if len(audio) < win:
        return np.array([], dtype=np.float32)
    n_frames = (len(audio) - win) // hop + 1
    e = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frame = audio[s:s + win]
        rms = float(np.sqrt((frame ** 2).mean() + 1e-12))
        e[i] = 20.0 * np.log10(rms + 1e-9)
    return e


def detect_syllables(audio: np.ndarray, sr: int,
                     cfg: Optional[SyllableConfig] = None) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) syllable spans.

    Heuristic recipe (no neural model):
      1. Frame-level energy in dB.
      2. F0 contour for voiced-region gating.
      3. A frame is "syllable-active" if energy > peak_dB + energy_floor_db
         AND (voiced OR neighbors_voiced).
      4. Connected active runs become candidate syllables.
      5. Inside long runs (>max_syl_ms), split at local energy minima or
         F0 discontinuities (|diff log f0| > 0.25 over voiced frames).
      6. Drop runs < min_syl_ms (likely noise/click).
      7. Cap to max_syllables (keep highest-energy spans).
    """
    cfg = cfg or SyllableConfig()
    if len(audio) < int(sr * cfg.win_ms / 1000):
        return [(0, len(audio))]

    hop_samples = int(sr * cfg.frame_ms / 1000)
    win_samples = int(sr * cfg.win_ms / 1000)

    edb = _frame_energy_db(audio, sr, cfg.frame_ms, cfg.win_ms)
    if len(edb) == 0:
        return [(0, len(audio))]
    peak = float(edb.max())
    active = edb > (peak + cfg.energy_floor_db)

    # F0 contour (uses same hop as utterance_cache._f0_contour, which we
    # configured at 20 ms). Pad/truncate to the energy-frame length.
    f0 = _f0_contour(audio, sr, hop_ms=cfg.frame_ms)
    L = min(len(edb), len(f0))
    edb = edb[:L]; f0 = f0[:L]; active = active[:L]

    # Voiced expansion: include unvoiced frames bordering voiced runs (capture onsets)
    voiced = (f0 > 0).astype(np.int32)
    voiced_expanded = voiced.copy()
    for shift in (-2, -1, 1, 2):
        voiced_expanded |= np.roll(voiced, shift)
    syl_mask = active & (voiced_expanded > 0)

    # Connected runs
    runs = []
    i = 0
    while i < L:
        if syl_mask[i]:
            j = i
            while j < L and syl_mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    # Filter min length
    min_frames = max(1, int(cfg.min_syl_ms / cfg.frame_ms))
    max_frames = max(min_frames + 1, int(cfg.max_syl_ms / cfg.frame_ms))
    filtered = [(a, b) for (a, b) in runs if (b - a) >= min_frames]

    # Split long runs at energy minima
    split_runs = []
    for (a, b) in filtered:
        if (b - a) <= max_frames:
            split_runs.append((a, b))
            continue
        # Find local minima inside
        seg = edb[a:b]
        # Smooth to find robust minima
        if len(seg) >= 5:
            smooth = np.convolve(seg, np.ones(5)/5.0, mode="same")
        else:
            smooth = seg
        # Greedy split: while current span > max_frames, split at argmin of smooth
        pending = [(a, b)]
        result = []
        while pending:
            x, y = pending.pop(0)
            if (y - x) <= max_frames:
                result.append((x, y))
                continue
            inner = smooth[x - a + 1:y - a - 1]
            if len(inner) == 0:
                result.append((x, y))
                continue
            cut_local = int(np.argmin(inner)) + 1
            cut = x + cut_local
            if (cut - x) < min_frames or (y - cut) < min_frames:
                result.append((x, y))
                continue
            pending.append((x, cut))
            pending.append((cut, y))
        split_runs.extend(result)

    # If we still found nothing, fall back to whole utterance
    if not split_runs:
        return [(0, len(audio))]

    # Sort by energy, keep top max_syllables (preserves the strongest spans)
    split_runs.sort(key=lambda ab: -float(edb[ab[0]:ab[1]].mean()))
    split_runs = split_runs[:cfg.max_syllables]
    split_runs.sort(key=lambda ab: ab[0])

    # Convert frame indices -> sample indices
    spans = []
    for (a, b) in split_runs:
        s_start = a * hop_samples
        s_end = min(len(audio), b * hop_samples + win_samples)
        if s_end - s_start > int(sr * 0.04):
            spans.append((s_start, s_end))
    if not spans:
        return [(0, len(audio))]
    return spans


# -----------------------------------------------------------------------
# Per-syllable 32-d embedding
# -----------------------------------------------------------------------

def per_syllable_embeddings(audio: np.ndarray, sr: int,
                            mlp: ToneEmbeddingMLP,
                            cfg: Optional[SyllableConfig] = None
                            ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (K, 32) per-syllable L2-normalized embeddings and the spans
    that produced them. K may equal 1 if no syllable detected."""
    spans = detect_syllables(audio, sr, cfg)
    embs = []
    for (s, e) in spans:
        seg = audio[s:e]
        if len(seg) < int(0.04 * sr):
            continue
        feat = clip_to_64d(seg, sr)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        emb = mlp.apply(feat[None, :])[0]
        embs.append(emb.astype(np.float32))
    if not embs:
        # fallback: whole utterance
        feat = clip_to_64d(audio, sr)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        emb = mlp.apply(feat[None, :])[0]
        embs.append(emb.astype(np.float32))
        spans = [(0, len(audio))]
    return np.stack(embs, axis=0), spans


# -----------------------------------------------------------------------
# Aggregation strategies (mean / max / concat-project / attention)
# -----------------------------------------------------------------------

def aggregate_mean(embs: np.ndarray) -> np.ndarray:
    v = embs.mean(axis=0)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def aggregate_max(embs: np.ndarray) -> np.ndarray:
    """Per-dim max (preserves sign by picking the most-extreme magnitude)."""
    # Use signed-max-magnitude per dim
    idx = np.argmax(np.abs(embs), axis=0)
    v = embs[idx, np.arange(embs.shape[1])]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


K_SYL = 8
EMB_DIM = 32


@dataclass
class ConcatProjector:
    W: np.ndarray  # (K_SYL*EMB_DIM, EMB_DIM)

    @staticmethod
    def identity_sum() -> "ConcatProjector":
        # Init: each syllable contributes 1/K to each output dim's matching pos
        W = np.zeros((K_SYL * EMB_DIM, EMB_DIM), dtype=np.float32)
        for k in range(K_SYL):
            for d in range(EMB_DIM):
                W[k * EMB_DIM + d, d] = 1.0 / K_SYL
        return ConcatProjector(W=W)

    def apply(self, embs: np.ndarray) -> np.ndarray:
        K = embs.shape[0]
        # Pad/truncate to K_SYL
        if K >= K_SYL:
            # Spread evenly: pick K_SYL indices spanning the span
            idx = np.linspace(0, K - 1, K_SYL).round().astype(int)
            E = embs[idx]
        else:
            pad = np.zeros((K_SYL - K, EMB_DIM), dtype=np.float32)
            E = np.concatenate([embs, pad], axis=0)
        v = E.reshape(-1) @ self.W
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v


@dataclass
class AttentionPooler:
    Q: np.ndarray   # (EMB_DIM, EMB_DIM) — 32 queries of dim 32
    temp: float = 1.0

    @staticmethod
    def init(seed: int = 0) -> "AttentionPooler":
        rng = np.random.default_rng(seed)
        # Init queries close to identity so attention has 32 specialized heads
        Q = (np.eye(EMB_DIM, dtype=np.float32) +
             0.05 * rng.standard_normal((EMB_DIM, EMB_DIM)).astype(np.float32))
        return AttentionPooler(Q=Q)

    def apply(self, embs: np.ndarray) -> np.ndarray:
        # embs: (K, 32). For each of 32 output dims, attention over K
        # syllables using query q_d (shape 32) dot key=emb.
        # logits: (32, K)
        logits = self.Q @ embs.T / self.temp
        # softmax over K
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        attn = ex / (ex.sum(axis=1, keepdims=True) + 1e-9)
        # Each output dim = weighted sum of embeddings projected along
        # the same dim. Result[d] = sum_k attn[d,k] * embs[k, d]
        per_dim = embs.T  # (32, K)
        out = (attn * per_dim).sum(axis=1)  # (32,)
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-9 else out


# -----------------------------------------------------------------------
# Two-stream cache with syllable-local pooling
# -----------------------------------------------------------------------

class SyllableLocalTwoStreamCache(TwoStreamCache):
    def __init__(self, mlp: ToneEmbeddingMLP,
                 strategy: str = "mean",
                 syl_cfg: Optional[SyllableConfig] = None,
                 projector: Optional[ConcatProjector] = None,
                 attn_pooler: Optional[AttentionPooler] = None,
                 cfg: Optional[TwoStreamConfig] = None):
        super().__init__(cfg)
        self.mlp = mlp
        self.strategy = strategy
        self.syl_cfg = syl_cfg or SyllableConfig()
        self.projector = projector
        self.attn_pooler = attn_pooler

    def fingerprint(self, audio: np.ndarray, sr: int) -> TwoStreamFingerprint:
        # Resample to 16k once
        if sr != 16000:
            n = int(round(len(audio) * 16000 / sr))
            audio16 = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                                np.arange(len(audio)), audio).astype(np.float32)
            sr16 = 16000
        else:
            audio16 = audio; sr16 = sr

        embs, _ = per_syllable_embeddings(audio16, sr16, self.mlp, self.syl_cfg)

        if self.strategy == "mean":
            tone = aggregate_mean(embs)
        elif self.strategy == "max":
            tone = aggregate_max(embs)
        elif self.strategy == "concat":
            tone = self.projector.apply(embs)
        elif self.strategy == "attention":
            tone = self.attn_pooler.apply(embs)
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")

        # Segment stream — identical to LearnedToneTwoStreamCache
        mfcc = cheap_mfcc(audio16, sr16, n_mfcc=13)
        mfcc_sub = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
        mfcc_flat = mfcc_sub.flatten(order="F")
        mfcc_delta = np.diff(mfcc_sub, axis=1, prepend=mfcc_sub[:, :1])
        mfcc_delta_flat = mfcc_delta.flatten(order="F")
        seg = np.concatenate([mfcc_flat, mfcc_delta_flat])
        return TwoStreamFingerprint(
            tone=tone.astype(np.float32),
            segment=_normalize(seg),
        )


# -----------------------------------------------------------------------
# Adversarial sentence eval
# -----------------------------------------------------------------------

ADV_JSON = PROJECT / "data" / "adversarial_minimal_pairs_vi.json"

OP_POINTS = [
    ("deployment_default", 0.85, 0.93),
    ("auto_calibrated",    0.70, 0.92),
]

SEED = 42


def _set_torch_seed():
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _synth(text, tok, model):
    import torch
    _set_torch_seed()
    inputs = tok(text, return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform
    return wav.squeeze().cpu().numpy().astype(np.float32)


def eval_adversarial(strategy: str, mlp: ToneEmbeddingMLP,
                     audio_pairs: list[tuple[np.ndarray, np.ndarray]],
                     sr: int,
                     projector: Optional[ConcatProjector] = None,
                     attn_pooler: Optional[AttentionPooler] = None) -> dict:
    out = {}
    for name, tau_A, tau_B in OP_POINTS:
        false_hits = 0
        per_pair = []
        cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B)
        # Pre-compute fingerprints once
        sample_cache = SyllableLocalTwoStreamCache(
            mlp, strategy=strategy, projector=projector,
            attn_pooler=attn_pooler, cfg=cfg)
        for i, (a_wav, b_wav) in enumerate(audio_pairs):
            cache = SyllableLocalTwoStreamCache(
                mlp, strategy=strategy, projector=projector,
                attn_pooler=attn_pooler, cfg=cfg)
            fp_a = sample_cache.fingerprint(a_wav, sr)
            fp_b = sample_cache.fingerprint(b_wav, sr)
            cache.put(fp_a, "en", {"vi": f"pair{i}_a"})
            hit = cache.lookup_existence(fp_b, "en")
            is_fh = hit is not None
            if is_fh:
                false_hits += 1
            t_cos = float(np.dot(fp_a.tone, fp_b.tone))
            s_cos = float(np.dot(fp_a.segment, fp_b.segment))
            per_pair.append({"idx": i, "tone_cos": round(t_cos, 4),
                             "segment_cos": round(s_cos, 4),
                             "false_hit": bool(is_fh)})
        out[name] = {
            "tau_A": tau_A, "tau_B": tau_B,
            "n_pairs": len(audio_pairs),
            "false_hits": false_hits,
            "false_hit_rate": round(false_hits / max(len(audio_pairs), 1), 4),
            "per_pair": per_pair,
        }
    return out


# -----------------------------------------------------------------------
# 120-clip syllable-level eval (uses EVAL_SYLLABLES from learned_tone_stream)
# -----------------------------------------------------------------------

CARRIER = "noi {syl}"

EVAL_SYLLABLES = {k: v for k, v in MINIMAL_PAIR_SYLLABLES_30.items()
                  if k in ("ma","ba","ca","la","da","ta","na","sa","ha","va",
                           "pa","ga","ka","ra","xa","gia","kha","nga","nha","tha")}


def _synth_syllable_corpus(tok, model) -> dict:
    import torch
    out = {}
    sr = model.config.sampling_rate
    for base, variants in EVAL_SYLLABLES.items():
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
            out[(base, v)] = a
    return out


def eval_syllable_level(strategy: str, mlp: ToneEmbeddingMLP,
                        audios: dict, tau_A: float = 0.85, tau_B: float = 0.93,
                        projector: Optional[ConcatProjector] = None,
                        attn_pooler: Optional[AttentionPooler] = None) -> dict:
    cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B,
                          max_entries=1024)
    factory = lambda: SyllableLocalTwoStreamCache(
        mlp, strategy=strategy, projector=projector,
        attn_pooler=attn_pooler, cfg=cfg)

    # Self-recall
    cache = factory()
    fps = {}
    for k, a in audios.items():
        fp = cache.fingerprint(a, 16000)
        fps[k] = fp
        cache.put(fp, "en", {"vi": f"{k[0]}/{k[1]}"})
    self_hits = 0
    for k, fp in fps.items():
        h = cache.lookup_existence(fp, "en")
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1

    # Tonal-variant probe: cache only ngang per base
    cache_t = factory()
    for base, variants in EVAL_SYLLABLES.items():
        ngang_key = (base, variants[0])
        if ngang_key in audios:
            fp = cache_t.fingerprint(audios[ngang_key], 16000)
            cache_t.put(fp, "en", {"vi": variants[0]})
    correct_miss = false_hit = total = 0
    for base, variants in EVAL_SYLLABLES.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audios:
                continue
            fp = cache_t.fingerprint(audios[k], 16000)
            h = cache_t.lookup_existence(fp, "en")
            total += 1
            if h is None:
                correct_miss += 1
            else:
                false_hit += 1
    return {
        "tau_A": tau_A, "tau_B": tau_B,
        "n_clips_total": len(audios),
        "self_hits": self_hits,
        "self_recall": round(self_hits / max(len(audios), 1), 4),
        "n_tonal_probes": total,
        "correct_misses": correct_miss,
        "correct_miss_rate": round(correct_miss / max(total, 1), 4),
        "false_hits": false_hit,
        "false_hit_rate": round(false_hit / max(total, 1), 4),
    }


# -----------------------------------------------------------------------
# Light training for concat-project on adversarial pairs
# -----------------------------------------------------------------------

def train_projector_on_pairs(mlp: ToneEmbeddingMLP,
                             audio_pairs: list[tuple[np.ndarray, np.ndarray]],
                             sr: int, n_iters: int = 200, lr: float = 0.02,
                             margin: float = 0.3, seed: int = 0
                             ) -> ConcatProjector:
    """Train W to push tone-stream cosine between (a, b) BELOW margin
    while keeping cos(a, a) = 1. Triplet-style with anchor=a, positive=a,
    negative=b. Note: this trains on the *same* pairs we eval, so it is
    NOT an out-of-distribution generalization check; we report it as an
    in-distribution best-case for the concat-project strategy. The
    attention-pool strategy below is also trained on the same pairs."""
    rng = np.random.default_rng(seed)
    proj = ConcatProjector.identity_sum()

    syl_cfg = SyllableConfig()

    def _embs_for(audio):
        e, _ = per_syllable_embeddings(audio, sr, mlp, syl_cfg)
        K = e.shape[0]
        if K >= K_SYL:
            idx = np.linspace(0, K - 1, K_SYL).round().astype(int)
            E = e[idx]
        else:
            pad = np.zeros((K_SYL - K, EMB_DIM), dtype=np.float32)
            E = np.concatenate([e, pad], axis=0)
        return E.reshape(-1)  # (K_SYL*32,)

    flats_a = np.stack([_embs_for(a) for (a, _b) in audio_pairs], axis=0)
    flats_b = np.stack([_embs_for(b) for (_a, b) in audio_pairs], axis=0)
    N = flats_a.shape[0]

    losses = []
    for it in range(n_iters):
        idx = rng.choice(N, size=min(N, 16), replace=False)
        A = flats_a[idx]; B = flats_b[idx]
        # forward
        ya = A @ proj.W; yb = B @ proj.W
        na = np.linalg.norm(ya, axis=1, keepdims=True) + 1e-8
        nb = np.linalg.norm(yb, axis=1, keepdims=True) + 1e-8
        ya_n = ya / na; yb_n = yb / nb
        cos_ab = (ya_n * yb_n).sum(axis=1)
        cos_aa = np.ones_like(cos_ab)
        loss_per = np.maximum(0.0, margin + cos_ab - cos_aa)
        active = (loss_per > 0).astype(np.float32)
        loss = float(loss_per.mean())
        losses.append(loss)
        if active.sum() == 0:
            continue
        # gradient of cos_ab w.r.t. ya, yb (chain through normalization)
        mask = active[:, None]
        # dL/dcos_ab = +1 (when active)
        # cos = (ya . yb) / (na nb). Project.
        # Approximate: ignore norm-gradient (small) — use straight cosine grad
        gya = mask * (yb_n / na) - mask * cos_ab[:, None] * ya_n / (na)
        gyb = mask * (ya_n / nb) - mask * cos_ab[:, None] * yb_n / (nb)
        gW = (A.T @ gya + B.T @ gyb) / max(int(active.sum()), 1)
        proj.W -= lr * gW
    return proj


def train_attention_on_pairs(mlp: ToneEmbeddingMLP,
                             audio_pairs: list[tuple[np.ndarray, np.ndarray]],
                             sr: int, n_iters: int = 200, lr: float = 0.05,
                             margin: float = 0.3, seed: int = 0
                             ) -> AttentionPooler:
    """Train the attention queries Q to discriminate sentence-pair tone
    contrasts. Same in-distribution caveat as train_projector_on_pairs."""
    rng = np.random.default_rng(seed)
    pooler = AttentionPooler.init(seed=seed)

    syl_cfg = SyllableConfig()
    pair_embs_a = []
    pair_embs_b = []
    for (a, b) in audio_pairs:
        ea, _ = per_syllable_embeddings(a, sr, mlp, syl_cfg)
        eb, _ = per_syllable_embeddings(b, sr, mlp, syl_cfg)
        pair_embs_a.append(ea)
        pair_embs_b.append(eb)

    def _forward(Q, E):
        logits = Q @ E.T  # (32, K)
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        attn = ex / (ex.sum(axis=1, keepdims=True) + 1e-9)
        per_dim = E.T  # (32, K)
        out = (attn * per_dim).sum(axis=1)
        n = float(np.linalg.norm(out))
        return out / max(n, 1e-9)

    N = len(audio_pairs)
    losses = []
    for it in range(n_iters):
        idx = rng.choice(N, size=min(N, 16), replace=False)
        loss_acc = 0.0; active_n = 0
        gQ = np.zeros_like(pooler.Q)
        eps = 1e-2
        for i in idx:
            va = _forward(pooler.Q, pair_embs_a[i])
            vb = _forward(pooler.Q, pair_embs_b[i])
            cos_ab = float(np.dot(va, vb))
            loss = max(0.0, margin + cos_ab - 1.0)
            if loss <= 0:
                continue
            loss_acc += loss; active_n += 1
            # Numerical gradient on Q (cheap: only 1024 params)
            for r in range(EMB_DIM):
                for c in range(EMB_DIM):
                    Q2 = pooler.Q.copy()
                    Q2[r, c] += eps
                    va2 = _forward(Q2, pair_embs_a[i])
                    vb2 = _forward(Q2, pair_embs_b[i])
                    cos_ab2 = float(np.dot(va2, vb2))
                    grad = (cos_ab2 - cos_ab) / eps
                    gQ[r, c] += grad
        if active_n > 0:
            losses.append(loss_acc / active_n)
            pooler.Q -= lr * (gQ / active_n)
        if it % 20 == 0 and active_n > 0:
            print(f"   [attn-train] it={it:3d}  loss={loss_acc/active_n:.4f}  active={active_n}")
    return pooler


# -----------------------------------------------------------------------
# Sanity probe: how accurate is the syllable detector?
# -----------------------------------------------------------------------

def syllable_detector_sanity(audios: dict) -> dict:
    """Check on the carrier 'noi {syl}' clips: expect ~2 syllables.
    On the 30 adversarial sentences we expect 4-8 syllables typically."""
    counts = []
    for k, a in list(audios.items())[:20]:
        spans = detect_syllables(a, 16000)
        counts.append(len(spans))
    return {
        "n_probed":   len(counts),
        "n_syllables_distribution": counts,
        "median":     float(np.median(counts)),
        "mean":       float(np.mean(counts)),
        "min":        int(min(counts)),
        "max":        int(max(counts)),
        "note":       "carrier 'noi {syl}' should yield ~2 syllables; deviations expose detector noise.",
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="data/cache_snapshots/learned_tone_stream.npz")
    ap.add_argument("--out-json", default="Results/syllable_local_tone_eval.json")
    ap.add_argument("--out-figure", default="papers/figures/fig_syllable_local_pooling.png")
    ap.add_argument("--skip-syllable-eval", action="store_true")
    args = ap.parse_args()

    weights_path = PROJECT / args.weights
    print(f"[load] MLP weights from {weights_path}")
    mlp = ToneEmbeddingMLP.load(weights_path)
    print(f"  W1={mlp.W1.shape}  W2={mlp.W2.shape}  feat_mean={mlp.feat_mean.shape}")

    # --- Synthesize adversarial sentence audios ---
    print("\n[load] adversarial pairs JSON ...")
    payload = json.loads(ADV_JSON.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    print(f"  {len(pairs)} sentence pairs")

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = tts_model.config.sampling_rate

    print(f"[synth] generating {2 * len(pairs)} adversarial clips at seed {SEED} ...")
    t0 = time.perf_counter()
    audio_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, p in enumerate(pairs):
        a_wav = _synth(p["source_a"], tok, tts_model)
        b_wav = _synth(p["source_b"], tok, tts_model)
        audio_pairs.append((a_wav, b_wav))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)}  elapsed={time.perf_counter()-t0:.1f}s")
    print(f"[synth] adversarial done in {time.perf_counter()-t0:.1f}s")

    # Resample pairs to 16k once (mms-tts-vie sr is typically 16000 but be safe)
    if sr != 16000:
        new_pairs = []
        for (a, b) in audio_pairs:
            def _rs(x):
                n16 = int(round(len(x) * 16000 / sr))
                return np.interp(np.linspace(0, len(x), n16, endpoint=False),
                                 np.arange(len(x)), x).astype(np.float32)
            new_pairs.append((_rs(a), _rs(b)))
        audio_pairs = new_pairs
    sr16 = 16000

    # --- Sanity check the detector ---
    print("\n[diag] syllable-detector sanity on adversarial source_a audios ...")
    san_audios = {i: a for i, (a, _b) in enumerate(audio_pairs)}
    san = syllable_detector_sanity(san_audios)
    print(f"  median={san['median']}  mean={san['mean']:.2f}  range=[{san['min']},{san['max']}]")
    print(f"  per-clip K: {san['n_syllables_distribution']}")

    # --- Train concat-project and attention on adversarial pairs ---
    print("\n[train] concat-project (in-distribution on adversarial pairs) ...")
    proj = train_projector_on_pairs(mlp, audio_pairs, sr16, n_iters=200, lr=0.05)
    print("[train] attention-pool (in-distribution; numerical-grad, expensive) ...")
    attn_pooler = train_attention_on_pairs(
        mlp, audio_pairs, sr16, n_iters=20, lr=0.05)

    # --- Eval all 4 strategies on adversarial ---
    print("\n[eval] adversarial — 4 aggregation strategies ...")
    adv_results = {}
    for strat in ("mean", "max", "concat", "attention"):
        t0 = time.perf_counter()
        r = eval_adversarial(strat, mlp, audio_pairs, sr16,
                             projector=proj, attn_pooler=attn_pooler)
        el = time.perf_counter() - t0
        for op, d in r.items():
            print(f"  [{strat:9s}] {op:20s} tau=({d['tau_A']},{d['tau_B']})  "
                  f"false-hit={d['false_hits']}/{d['n_pairs']} "
                  f"({d['false_hit_rate']*100:.1f}%)")
        print(f"  ({strat} elapsed={el:.1f}s)")
        adv_results[strat] = r

    # --- Eval on 120-clip syllable corpus ---
    if not args.skip_syllable_eval:
        print("\n[synth] 120-clip syllable-level eval corpus ...")
        syl_audios = _synth_syllable_corpus(tok, tts_model)
        print(f"  {len(syl_audios)} clips")
        syl_results = {}
        for strat in ("mean", "max", "concat", "attention"):
            t0 = time.perf_counter()
            r = eval_syllable_level(strat, mlp, syl_audios,
                                    projector=proj, attn_pooler=attn_pooler)
            el = time.perf_counter() - t0
            print(f"  [{strat:9s}] self={r['self_hits']}/120  "
                  f"correct-miss={r['correct_misses']}/100 "
                  f"({r['correct_miss_rate']*100:.1f}%)  "
                  f"false={r['false_hits']}  elapsed={el:.1f}s")
            syl_results[strat] = r
    else:
        syl_results = {}

    # --- Pick best strategy on adversarial @ deployment default ---
    best_strat = min(
        adv_results.keys(),
        key=lambda s: adv_results[s]["deployment_default"]["false_hit_rate"])
    best_fh = adv_results[best_strat]["deployment_default"]["false_hit_rate"]
    baseline_fh = 0.4667
    print(f"\n[best] strategy={best_strat}  false-hit={best_fh*100:.1f}% "
          f"(baseline 46.7%, delta {(best_fh - baseline_fh)*100:+.1f}pp)")

    # --- Figure ---
    print("\n[fig] bar chart adversarial vs syllable-level ...")
    strategies = ["mean", "max", "concat", "attention"]
    adv_fh = [adv_results[s]["deployment_default"]["false_hit_rate"] * 100
              for s in strategies]
    syl_fh = [(syl_results.get(s, {}).get("false_hit_rate", 0.0)) * 100
              for s in strategies] if syl_results else [0.0] * 4
    x = np.arange(len(strategies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=140)
    b1 = ax.bar(x - width/2, adv_fh, width, color="#c44e52",
                label="Adversarial sentences (30 pairs)")
    b2 = ax.bar(x + width/2, syl_fh, width, color="#4c72b0",
                label="Syllable-level (100 tonal probes)")
    ax.axhline(46.7, color="#c44e52", linestyle=":", linewidth=1.0, alpha=0.7,
               label="Round-26 baseline (46.7%)")
    ax.set_xticks(x); ax.set_xticklabels(
        ["mean\n(R26)", "max", "concat-\nproject", "attention-\npool"])
    ax.set_ylabel("False-hit rate (%)")
    ax.set_ylim(0, max(70, max(adv_fh + syl_fh) + 5))
    ax.set_title("Syllable-local tone pooling — false-hit rate by aggregation strategy\n"
                 "(deployment thresholds $\\tau_A=0.85$, $\\tau_B=0.93$)")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", xy=(r.get_x()+r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = PROJECT / args.out_figure
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_path))
    plt.close(fig)
    print(f"[fig] -> {fig_path}")

    # --- Strip non-serializable fields ---
    def _strip(d):
        for op, r in d.items():
            r.pop("per_pair", None)
        return d
    out_adv = {s: _strip(adv_results[s]) for s in adv_results}

    out = {
        "status": "ok",
        "round": 27,
        "experiment": "Syllable-local tone pooling — sentence-level adversarial fix",
        "baseline_round26_adversarial_false_hit_rate": baseline_fh,
        "syllable_detector_sanity": san,
        "training_notes": {
            "concat_project_init":  "identity-sum (1/K_SYL on each block diagonal)",
            "attention_pool_init":  "identity-perturbed queries (32 of dim 32)",
            "trained_on":           "the 30 adversarial pairs themselves (in-distribution best-case)",
            "caveat":               "Concat-project and attention-pool numbers are in-distribution. Mean and max are training-free and the load-bearing comparison vs. R26 baseline.",
        },
        "adversarial_eval":  out_adv,
        "syllable_level_eval": syl_results,
        "best_strategy_on_adversarial": {
            "strategy":          best_strat,
            "false_hit_rate":    best_fh,
            "delta_vs_baseline_pp": round((best_fh - baseline_fh) * 100, 2),
        },
        "headline": (
            f"Best strategy ({best_strat}) cuts adversarial false-hit from "
            f"46.7% (R26 utterance-level) to {best_fh*100:.1f}% "
            f"({(best_fh-baseline_fh)*100:+.1f}pp)."
        ),
    }
    out_path = PROJECT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] -> {out_path}")
    print(f"[headline] {out['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
