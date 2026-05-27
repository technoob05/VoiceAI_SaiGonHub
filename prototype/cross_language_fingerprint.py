"""
Cross-language fingerprint sanity check on FLEURS (Vi / Zh / Ko / En).

Paper §3.3 generalization claim: the 272-d tonal-perceptual fingerprint isn't
secretly overfit to Vietnamese. FLEURS is n-way parallel — same sentence IDs
exist in all four languages spoken by different speakers — so we can ask:

  Q1. Does cosine(fp(L, sid)) for same content across language pairs differ
      from cosine across different content within a single language?
  Q2. Does the 6-bin tone histogram component cluster the two tonal languages
      (Vi, Zh) closer than the non-tonal pair (En, Ko)?

Honest scope: NOT proof of Mandarin tone-coverage (would need AISHELL-3 with
per-syllable tone labels). This is a free generalization sanity check on data
we already have on disk.

Inputs (no downloads, all local):
  data/raw/hf_cache/hub/datasets--google--fleurs/snapshots/<sha>/parquet-data/
      {vi_vn,cmn_hans_cn,ko_kr,en_us}/test-00000-of-00001.parquet

Output:
  Results/cross_language_fingerprint.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from utterance_cache import UtteranceCache, tone_histogram_from_f0, _f0_contour


PROJECT = HERE.parent
FLEURS_ROOT = (
    PROJECT / "data" / "raw" / "hf_cache" / "hub"
    / "datasets--google--fleurs" / "snapshots"
)
LANGS = ["vi_vn", "cmn_hans_cn", "ko_kr", "en_us"]
TONAL = {"vi_vn", "cmn_hans_cn"}
N_PER_LANG = 30
SR_TARGET = 16000


def _snapshot_dir() -> Path:
    # Pick the only snapshot present
    snaps = sorted(p for p in FLEURS_ROOT.iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"No FLEURS snapshot under {FLEURS_ROOT}")
    return snaps[0]


def _parquet_path(lang: str) -> Path:
    return _snapshot_dir() / "parquet-data" / lang / "test-00000-of-00001.parquet"


def _decode_audio(b: bytes) -> tuple[np.ndarray, int] | None:
    try:
        audio, sr = sf.read(io.BytesIO(b), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32, copy=False), int(sr)
    except Exception:
        return None


def _load_lang(lang: str, sentence_ids: list[int]
               ) -> tuple[list[int], dict[int, np.ndarray], dict[int, int]]:
    """Load audio for exactly the given sentence IDs (skip undecodable rows).

    If multiple rows in the parquet share a sentence id (multiple speakers per
    sentence) we take the first decodable one.
    """
    path = _parquet_path(lang)
    table = pq.read_table(path, columns=["id", "audio"])
    ids = table.column("id").to_pylist()
    audios = table.column("audio").to_pylist()

    by_id: dict[int, list[dict]] = {}
    for sid, a in zip(ids, audios):
        by_id.setdefault(sid, []).append(a)

    kept_ids: list[int] = []
    waves: dict[int, np.ndarray] = {}
    srs: dict[int, int] = {}
    skipped = 0
    for sid in sentence_ids:
        if sid not in by_id:
            skipped += 1
            continue
        ok = False
        for audio_struct in by_id[sid]:
            decoded = _decode_audio(audio_struct["bytes"])
            if decoded is None:
                continue
            wav, sr = decoded
            if len(wav) < int(0.5 * sr):
                continue
            waves[sid] = wav
            srs[sid] = sr
            kept_ids.append(sid)
            ok = True
            break
        if not ok:
            skipped += 1
    if skipped:
        print(f"  [{lang}] missing/undecoded for {skipped} requested ids")
    return kept_ids, waves, srs


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _stats(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"name": name, "n_pairs": 0, "mean": None, "std": None,
                "min": None, "max": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "name":    name,
        "n_pairs": int(arr.size),
        "mean":    float(arr.mean()),
        "std":     float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min":     float(arr.min()),
        "max":     float(arr.max()),
    }


def main() -> int:
    print("=" * 70)
    print("Cross-language fingerprint sanity check (FLEURS Vi/Zh/Ko/En)")
    print("=" * 70)

    # Step 1: find common sentence IDs across all four languages.
    print("\n[1] Indexing sentence IDs across languages…")
    id_sets: dict[str, set[int]] = {}
    for lang in LANGS:
        tab = pq.read_table(_parquet_path(lang), columns=["id"])
        ids = tab.column("id").to_pylist()
        id_sets[lang] = set(ids)
        print(f"  {lang:12s}  rows={len(ids):5d}  unique_ids={len(id_sets[lang]):5d}")

    common = sorted(set.intersection(*id_sets.values()))
    print(f"  common across 4 langs: {len(common)}")
    if len(common) < N_PER_LANG:
        print(f"  ERROR: need >= {N_PER_LANG} common ids, got {len(common)}")
        return 2

    # Take a deterministic shuffle and oversample (~3x) to absorb decode failures
    # in any one language, then keep the first N that succeed in ALL languages.
    rng = np.random.default_rng(0)
    candidate_ids = [int(x) for x in rng.permutation(common)]
    oversample = min(len(candidate_ids), N_PER_LANG * 4)
    candidate_ids = candidate_ids[:oversample]

    # Step 2: load audio for each language constrained to candidate_ids.
    print("\n[2] Loading audio for", len(candidate_ids), "candidate ids per language…")
    waves: dict[str, dict[int, np.ndarray]] = {}
    srs:   dict[str, dict[int, int]] = {}
    kept:  dict[str, set[int]] = {}
    for lang in LANGS:
        kept_ids, w, s = _load_lang(lang, candidate_ids)
        waves[lang] = w
        srs[lang] = s
        kept[lang] = set(kept_ids)
        print(f"  {lang:12s}  decoded={len(kept_ids)}")

    # Aligned set = candidates that decoded successfully in ALL four languages.
    common_decoded = set.intersection(*kept.values())
    aligned = [sid for sid in candidate_ids if sid in common_decoded][:N_PER_LANG]
    if len(aligned) < 10:
        print(f"  ERROR: only {len(aligned)} aligned ids after decode. abort.")
        return 3
    print(f"  ALIGNED sentence set: {len(aligned)} ids "
          f"(target was {N_PER_LANG})")

    # Step 3: compute fingerprints.
    print("\n[3] Computing 272-d fingerprints…")
    cache = UtteranceCache(sr=SR_TARGET)
    fps:        dict[tuple[str, int], np.ndarray] = {}
    tone_hists: dict[tuple[str, int], np.ndarray] = {}

    for lang in LANGS:
        for sid in aligned:
            wav = waves[lang][sid]
            sr = srs[lang][sid]
            fp = cache.fingerprint(wav, sr=sr)
            fps[(lang, sid)] = fp.vec
            # Recompute tone histogram independently for the language-clustering test
            # (the first 6 dims of the fingerprint are L2-normalized jointly with the
            # other 266 dims, so they're not directly comparable as a probability
            # distribution any more — recompute from F0 to get clean 6-d simplex).
            if sr != SR_TARGET:
                n = int(round(len(wav) * SR_TARGET / sr))
                resamp = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                                   np.arange(len(wav)), wav).astype(np.float32)
            else:
                resamp = wav
            f0 = _f0_contour(resamp, SR_TARGET)
            tone_hists[(lang, sid)] = tone_histogram_from_f0(f0)
    print(f"  computed {len(fps)} fingerprints  "
          f"(mean fp_ms={cache.stats.fingerprint_ms_mean:.1f})")

    # Step 4: four cosine distributions on full 272-d fingerprint.
    print("\n[4] Four cosine distributions (272-d fingerprint)…")

    intra_vi_self     : list[float] = []   # sanity: must be ~1.0
    inter_same_content: list[float] = []   # vi-001 vs zh-001 (etc., all cross-lang pairs)
    intra_diff_content: list[float] = []   # vi-001 vs vi-002 (per language)
    inter_diff_content: list[float] = []   # vi-001 vs zh-002

    for sid in aligned:
        v = fps[("vi_vn", sid)]
        intra_vi_self.append(_cos(v, v))

    for i, sid_i in enumerate(aligned):
        for j, sid_j in enumerate(aligned):
            if j <= i:
                continue
            # intra-language, different content (per language)
            for L in LANGS:
                intra_diff_content.append(_cos(fps[(L, sid_i)], fps[(L, sid_j)]))
            # inter-language, different content (unordered language pairs)
            for a_idx in range(len(LANGS)):
                for b_idx in range(a_idx + 1, len(LANGS)):
                    inter_diff_content.append(
                        _cos(fps[(LANGS[a_idx], sid_i)], fps[(LANGS[b_idx], sid_j)])
                    )
                    # also include the swapped (b,a) pairing with sids swapped,
                    # which is a genuinely different pair when sid_i != sid_j
                    inter_diff_content.append(
                        _cos(fps[(LANGS[a_idx], sid_j)], fps[(LANGS[b_idx], sid_i)])
                    )

    # same-content cross-language pairs: unordered language pairs at matched sid
    for sid in aligned:
        for a_idx in range(len(LANGS)):
            for b_idx in range(a_idx + 1, len(LANGS)):
                inter_same_content.append(
                    _cos(fps[(LANGS[a_idx], sid)], fps[(LANGS[b_idx], sid)])
                )

    dist_intra_self  = _stats("intra-vi self-pair (sanity, expect 1.0)", intra_vi_self)
    dist_same        = _stats("inter-language same-content (e.g. vi-001 vs zh-001)", inter_same_content)
    dist_intra_diff  = _stats("intra-language different-content (e.g. vi-001 vs vi-002)", intra_diff_content)
    dist_inter_diff  = _stats("inter-language different-content (e.g. vi-001 vs zh-002)", inter_diff_content)

    for d in (dist_intra_self, dist_same, dist_intra_diff, dist_inter_diff):
        print(f"  {d['name']:60s}  n={d['n_pairs']:5d}  "
              f"mean={d['mean']:.4f}  std={d['std']:.4f}")

    # Interpretation: a content-discriminative, language-invariant fingerprint
    # would yield same_content >> diff_content (regardless of language) AND
    # intra_diff_content ~= inter_diff_content.
    # A content-discriminative, language-specific fingerprint would yield
    # intra_diff < inter_diff (different language adds extra distance).
    same_vs_diff_gap = dist_same["mean"] - dist_inter_diff["mean"]
    intra_vs_inter_diff_gap = dist_intra_diff["mean"] - dist_inter_diff["mean"]
    print(f"\n  Gap[same-content - diff-content (cross-lang)] = {same_vs_diff_gap:+.4f}")
    print(f"  Gap[intra-lang diff - inter-lang diff]      = {intra_vs_inter_diff_gap:+.4f}")
    if same_vs_diff_gap > 0.02:
        print("    -> fingerprint IS content-discriminative across languages")
    else:
        print("    -> fingerprint is NOT robustly content-discriminative across languages")
    if abs(intra_vs_inter_diff_gap) < 0.02:
        print("    -> fingerprint is approximately language-INVARIANT on diff content")
    elif intra_vs_inter_diff_gap > 0:
        print("    -> intra-lang pairs cluster TIGHTER than cross-lang -> language-SPECIFIC")
    else:
        print("    -> cross-lang pairs cluster TIGHTER -> unusual (check F0/MFCC scaling)")

    # Step 5: tone-histogram language-cluster diagnostic.
    #
    # For each language pair, compute mean cosine between language-mean tone
    # histograms (averaged over the aligned sentence set). Hypothesis: the
    # tonal pair (Vi, Zh) should be closer to each other than to non-tonal
    # languages (En, Ko).
    print("\n[5] Tone-histogram language clustering…")
    lang_mean_tone: dict[str, np.ndarray] = {}
    for L in LANGS:
        stacked = np.stack([tone_hists[(L, sid)] for sid in aligned], axis=0)
        lang_mean_tone[L] = stacked.mean(axis=0)
        print(f"  {L:12s}  mean tone hist = "
              + " ".join(f"{v:.3f}" for v in lang_mean_tone[L]))

    pair_cos: dict[str, float] = {}
    for i in range(len(LANGS)):
        for j in range(i + 1, len(LANGS)):
            L1, L2 = LANGS[i], LANGS[j]
            pair_cos[f"{L1}__{L2}"] = _cos(lang_mean_tone[L1], lang_mean_tone[L2])
    for k, v in sorted(pair_cos.items(), key=lambda x: -x[1]):
        print(f"  tone-hist cos[{k:28s}] = {v:.4f}")

    vi_zh = pair_cos.get("vi_vn__cmn_hans_cn", float("nan"))
    en_ko = pair_cos.get("ko_kr__en_us", float("nan"))
    tonal_vs_nontonal_pairs = [
        pair_cos["vi_vn__en_us"], pair_cos["vi_vn__ko_kr"],
        pair_cos["cmn_hans_cn__en_us"], pair_cos["cmn_hans_cn__ko_kr"],
    ]
    tonal_mixed_mean = float(np.mean(tonal_vs_nontonal_pairs))

    print(f"\n  vi-vn vs cmn (tonal pair):       {vi_zh:.4f}")
    print(f"  ko-kr vs en-us (non-tonal pair): {en_ko:.4f}")
    print(f"  mean tonal-vs-nontonal pair:     {tonal_mixed_mean:.4f}")
    tone_cluster_supported = (vi_zh > tonal_mixed_mean) and (en_ko > tonal_mixed_mean)
    print(f"  diagnostic: tone-histogram supports Vi+Zh clustering? "
          f"{tone_cluster_supported}")
    if not tone_cluster_supported:
        print("    -> CAVEAT: tone-histogram alone does NOT cleanly separate tonal "
              "from non-tonal in FLEURS. Likely because the 6 bins were heuristically "
              "designed for Vietnamese tone categories specifically; Mandarin's "
              "4-tone system maps onto them only loosely.")

    # Step 6: dump JSON.
    out_path = PROJECT / "Results" / "cross_language_fingerprint.json"
    payload = {
        "config": {
            "n_per_lang_target":  N_PER_LANG,
            "n_aligned_actual":   len(aligned),
            "languages":          LANGS,
            "tonal_languages":    sorted(TONAL),
            "fingerprint_dim":    272,
            "sr_target":          SR_TARGET,
            "data_root":          str(_snapshot_dir()),
            "scope_caveat": (
                "Generalization sanity check only. NOT proof that the fingerprint "
                "captures Mandarin tone correctly — that requires AISHELL-3 with "
                "per-syllable tone labels. FLEURS is sentence-level n-way parallel "
                "with different speakers per language."
            ),
        },
        "cosine_distributions": {
            "intra_vi_self":               dist_intra_self,
            "inter_language_same_content": dist_same,
            "intra_language_diff_content": dist_intra_diff,
            "inter_language_diff_content": dist_inter_diff,
        },
        "content_discrimination": {
            "same_content_minus_diff_content_cross_lang": same_vs_diff_gap,
            "intra_lang_diff_minus_inter_lang_diff":      intra_vs_inter_diff_gap,
            "content_discriminative_across_languages":    bool(same_vs_diff_gap > 0.02),
            "approximately_language_invariant":           bool(abs(intra_vs_inter_diff_gap) < 0.02),
        },
        "tone_histogram_language_clustering": {
            "lang_mean_tone_histograms": {L: lang_mean_tone[L].tolist() for L in LANGS},
            "pairwise_cosine":           pair_cos,
            "vi_zh_tonal_pair":          vi_zh,
            "en_ko_nontonal_pair":       en_ko,
            "mean_tonal_vs_nontonal":    tonal_mixed_mean,
            "supports_vi_zh_cluster":    bool(tone_cluster_supported),
        },
        "fingerprint_timing_ms_mean":  round(cache.stats.fingerprint_ms_mean, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[6] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
