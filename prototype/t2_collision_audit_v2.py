"""
T2 v2 collision audit — re-measures the diacritic-conflation false-hit rate
of the two-stream AND-gated T2 cache (translation_memory_v2) on the same
3,000-VIVOS-train sample used by t2_collision_audit.py.

Protocol (parallel to v1, swap rule):
  1. Same VIVOS-train uniques, same seed-42 sub-sample of 3,000.
  2. For each transcript compute both fp_S (stripped) and fp_P (preserving).
  3. For every unordered pair (s_i, s_j), flag a v2 collision when
        cos_S(s_i, s_j) >= tau_S  AND
        cos_P(s_i, s_j) >= tau_P  AND
        Lev(norm_i, norm_j) > lev_frac * max(|s_i|, |s_j|).
     Drop the row if the AND-gate vetoes (cos_S high but cos_P low) —
     this is the documented mitigation in §6.6.
  4. Also measure paraphrase recall on a 20-pair true-paraphrase probe set
     (same meaning, different surface).

Outputs Results/t2_collision_audit_v2.json with v1↔v2 deltas.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

from translation_memory_v2 import (  # noqa: E402
    text_fingerprint_stripped,
    text_fingerprint_preserving,
    _normalize_preserving,
    _normalize_stripped,
)

PROJECT     = HERE.parent
VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

TAU_S    = 0.80
TAU_P    = 0.85
LEV_FRAC = 0.30


_PUNCT_RE = re.compile(r"[^\w\s\xC0-ỹ]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def diacritic_preserving_normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def diacritic_stripped(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV
    def levenshtein(a: str, b: str) -> int:
        return _RF_LEV.distance(a, b)
except Exception:
    def levenshtein(a: str, b: str) -> int:
        if a == b: return 0
        if len(a) < len(b): a, b = b, a
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                ins  = cur[j - 1] + 1
                dele = prev[j] + 1
                sub  = prev[j - 1] + (0 if ca == cb else 1)
                cur[j] = min(ins, dele, sub)
            prev = cur
        return prev[-1]


def _load_vivos_unique_transcripts() -> list[str]:
    pf = VIVOS_TRAIN / "prompts.txt"
    if not pf.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in pf.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        _, _, text = line.partition(" ")
        text = text.strip()
        if not text: continue
        key = diacritic_preserving_normalize(text)
        if key and key not in seen:
            seen.add(key); out.append(text)
    return out


# ---------------- True-paraphrase probe set (20 pairs) ----------------

PARAPHRASE_PAIRS = [
    # (stored, query)  — same meaning + same diacritics, surface variation only.
    ("an toàn lao động",            "An Toàn Lao Động"),
    ("an toàn lao động",            "an toàn lao động!"),
    ("an toàn lao động",            "  an toàn lao động  "),
    ("dừng dây chuyền ngay",        "Dừng dây chuyền ngay."),
    ("dừng dây chuyền ngay",        "DỪNG DÂY CHUYỀN NGAY"),
    ("máy đang chạy",                "Máy đang chạy."),
    ("máy đang chạy",                "máy đang chạy!!!"),
    ("tôi muốn uống nước",           "Tôi muốn uống nước"),
    ("tôi muốn uống nước",           "tôi muốn uống nước."),
    ("cảm ơn rất nhiều",             "Cảm ơn rất nhiều!"),
    ("cảm ơn rất nhiều",             "CẢM ƠN RẤT NHIỀU"),
    ("xin chào các bạn",             "Xin chào các bạn."),
    ("xin chào các bạn",             "xin chào, các bạn"),
    ("đèn đỏ phía trước",            "Đèn đỏ phía trước!"),
    ("rẽ trái ở ngã tư",             "Rẽ trái ở ngã tư."),
    ("ngày mai trời mưa",            "Ngày mai trời mưa."),
    ("tôi đến từ hà nội",            "Tôi đến từ Hà Nội."),
    ("tôi yêu việt nam",             "Tôi yêu Việt Nam!"),
    ("bạn có khỏe không",            "Bạn có khỏe không?"),
    ("cảm ơn rất nhiều",             "cảm ơn rất nhiều."),
]


def measure_paraphrase_recall(tau_p: float, tau_s: float) -> dict:
    """For each pair (stored, query), import v2 lookup logic inline."""
    hits = 0
    misses_detail = []
    for stored, query in PARAPHRASE_PAIRS:
        s_p = text_fingerprint_preserving(stored)
        s_s = text_fingerprint_stripped(stored)
        q_p = text_fingerprint_preserving(query)
        q_s = text_fingerprint_stripped(query)
        cp = float(np.dot(s_p, q_p))
        cs = float(np.dot(s_s, q_s))
        if cp >= tau_p and cs >= tau_s:
            hits += 1
        else:
            misses_detail.append({
                "stored": stored, "query": query,
                "cos_p": cp, "cos_s": cs,
                "veto_by": ("P" if cp < tau_p else "") + ("S" if cs < tau_s else ""),
            })
    return {"n_pairs": len(PARAPHRASE_PAIRS),
            "hits": hits, "misses": len(PARAPHRASE_PAIRS) - hits,
            "recall": hits / len(PARAPHRASE_PAIRS),
            "misses_detail": misses_detail}


# ---------------- Audit ----------------

def audit(corpus: list[str], tau_s: float, tau_p: float,
          lev_frac: float, chunk: int = 512) -> dict:
    n = len(corpus)
    print(f"[fp] computing {n} stripped + preserving fingerprints ...", flush=True)
    t0 = time.perf_counter()
    fps_s = np.stack([text_fingerprint_stripped(t) for t in corpus]).astype(np.float32)
    fps_p = np.stack([text_fingerprint_preserving(t) for t in corpus]).astype(np.float32)
    norm_orig = [diacritic_preserving_normalize(t) for t in corpus]
    print(f"  done in {time.perf_counter()-t0:.2f}s "
          f"shape_s={fps_s.shape} shape_p={fps_p.shape}", flush=True)

    total_pairs = n * (n - 1) // 2
    lens = np.asarray([len(s) for s in norm_orig], dtype=np.int32)

    collisions: list[dict] = []
    n_s_pass = 0
    n_both_pass = 0
    n_lev_calls = 0
    t0 = time.perf_counter()
    for r0 in range(0, n, chunk):
        r1 = min(r0 + chunk, n)
        block_s = fps_s[r0:r1] @ fps_s.T
        block_p = fps_p[r0:r1] @ fps_p.T
        for li, i in enumerate(range(r0, r1)):
            row_s = block_s[li]
            row_p = block_p[li]
            j_idx = np.flatnonzero(row_s >= tau_s)
            j_idx = j_idx[j_idx > i]
            n_s_pass += int(j_idx.size)
            if j_idx.size == 0:
                continue
            # AND-gate: filter to those also passing tau_p
            p_vals = row_p[j_idx]
            j_idx_both = j_idx[p_vals >= tau_p]
            n_both_pass += int(j_idx_both.size)
            if j_idx_both.size == 0:
                continue
            a = norm_orig[i]
            la = lens[i]
            if not a:
                continue
            jb = j_idx_both
            lb = lens[jb]
            maxL = np.maximum(la, lb)
            absdiff = np.abs(la - lb)
            auto_mask = absdiff > (lev_frac * maxL)
            auto_idx     = jb[auto_mask]
            need_lev_idx = jb[~auto_mask]

            for j in auto_idx.tolist():
                b = norm_orig[j]
                if not b: continue
                L = int(max(la, lens[j]))
                d = int(abs(la - lens[j]))
                collisions.append({
                    "i": int(i), "j": int(j),
                    "vi_i": corpus[i], "vi_j": corpus[j],
                    "stripped_i": diacritic_stripped(corpus[i]),
                    "stripped_j": diacritic_stripped(corpus[j]),
                    "cos_s": float(row_s[j]),
                    "cos_p": float(row_p[j]),
                    "lev": d, "max_len": L,
                    "lev_norm": float(d / L) if L else 0.0,
                    "via": "length_diff",
                })

            for j in need_lev_idx.tolist():
                b = norm_orig[j]
                if not b: continue
                L = int(max(la, lens[j]))
                d = levenshtein(a, b)
                n_lev_calls += 1
                if d > lev_frac * L:
                    collisions.append({
                        "i": int(i), "j": int(j),
                        "vi_i": corpus[i], "vi_j": corpus[j],
                        "stripped_i": diacritic_stripped(corpus[i]),
                        "stripped_j": diacritic_stripped(corpus[j]),
                        "cos_s": float(row_s[j]),
                        "cos_p": float(row_p[j]),
                        "lev": int(d), "max_len": L,
                        "lev_norm": float(d / L) if L else 0.0,
                        "via": "lev",
                    })
        if r0 % (chunk * 4) == 0 or r1 == n:
            elapsed = time.perf_counter() - t0
            done = r1 / n
            eta = elapsed / max(done, 1e-6) - elapsed
            print(f"  row {r1}/{n} ({done*100:.1f}%) "
                  f"S_pass={n_s_pass} both_pass={n_both_pass} "
                  f"lev_calls={n_lev_calls} coll={len(collisions)} "
                  f"elapsed={elapsed:.1f}s eta={eta:.0f}s", flush=True)

    print(f"  scan done in {time.perf_counter()-t0:.1f}s", flush=True)
    collisions.sort(key=lambda r: (-(r["cos_s"] + r["cos_p"]), -r["lev_norm"]))
    n_coll = len(collisions)
    rate = n_coll / total_pairs if total_pairs else 0.0

    return {
        "config": {
            "tau_s": tau_s, "tau_p": tau_p, "lev_frac": lev_frac,
            "n_corpus": n, "total_pairs": total_pairs,
            "n_above_tau_s": int(n_s_pass),
            "n_above_both":  int(n_both_pass),
            "n_collisions":  n_coll,
            "collision_rate": rate,
        },
        "worst_10": collisions[:10],
        "all_collisions": collisions[:200],  # truncate to keep file small
        "n_collisions_total": n_coll,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-s", type=float, default=TAU_S)
    ap.add_argument("--tau-p", type=float, default=TAU_P)
    ap.add_argument("--lev-frac", type=float, default=LEV_FRAC)
    ap.add_argument("--max-corpus", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="Results/t2_collision_audit_v2.json")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("[corpus] loading VIVOS train prompts ...", flush=True)
    corpus = _load_vivos_unique_transcripts()
    source = "vivos_train_prompts"
    if not corpus:
        print("  ERROR: VIVOS prompts.txt not found at "
              f"{VIVOS_TRAIN/'prompts.txt'}", flush=True)
        return 2
    print(f"  loaded {len(corpus)} unique transcripts", flush=True)

    if args.max_corpus and len(corpus) > args.max_corpus:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(corpus), args.max_corpus, replace=False)
        corpus = [corpus[i] for i in sorted(idx)]
        print(f"  sub-sampled to {len(corpus)} (seed={args.seed})", flush=True)

    result = audit(corpus, tau_s=args.tau_s, tau_p=args.tau_p,
                   lev_frac=args.lev_frac)
    result["source"] = source

    print("\n[paraphrase] measuring true-paraphrase recall ...", flush=True)
    recall = measure_paraphrase_recall(tau_p=args.tau_p, tau_s=args.tau_s)
    result["paraphrase_recall"] = recall

    # v1 baseline for delta reporting
    v1_path = PROJECT / "Results" / "t2_collision_audit.json"
    v1_rate = None
    if v1_path.exists():
        try:
            v1_data = json.loads(v1_path.read_text(encoding="utf-8"))
            v1_rate = v1_data["config"]["collision_rate"]
            result["v1_collision_rate"]      = v1_rate
            result["delta_collision_rate"]   = v1_rate - result["config"]["collision_rate"]
            result["reduction_factor"]       = (v1_rate / result["config"]["collision_rate"]
                                                 if result["config"]["collision_rate"] > 0
                                                 else float("inf"))
        except Exception:
            pass

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] wrote {out_path}")

    print("\n" + "=" * 78)
    print(f"T2 v2 AUDIT — tau_S={args.tau_s} tau_P={args.tau_p} "
          f"lev>={args.lev_frac}*max_len")
    print("=" * 78)
    c = result["config"]
    print(f"  corpus size                : {c['n_corpus']:>10}")
    print(f"  total unordered pairs      : {c['total_pairs']:>10}")
    print(f"  pairs above cos_S          : {c['n_above_tau_s']:>10}")
    print(f"  pairs above BOTH gates     : {c['n_above_both']:>10}")
    print(f"  collision pairs (AND-gated): {c['n_collisions']:>10}")
    print(f"  collision rate v2          : {c['collision_rate']*100:>10.4f}%")
    if v1_rate is not None:
        print(f"  collision rate v1 baseline : {v1_rate*100:>10.4f}%")
        print(f"  reduction (v1 / v2)        : {result['reduction_factor']:>10.1f}×")

    r = result["paraphrase_recall"]
    print(f"\n  paraphrase recall          : {r['hits']}/{r['n_pairs']} "
          f"({r['recall']*100:.1f}%)")
    if r["misses_detail"]:
        print("  paraphrase misses:")
        for m in r["misses_detail"]:
            print(f"    cosP={m['cos_p']:.3f} cosS={m['cos_s']:.3f} "
                  f"veto={m['veto_by']:<2}  {m['stored']!r} <- {m['query']!r}")

    if result["worst_10"]:
        print("\n  worst-10 surviving collisions (cos_S, cos_P, lev_norm):")
        for r2 in result["worst_10"]:
            print(f"    cs={r2['cos_s']:.3f} cp={r2['cos_p']:.3f} "
                  f"lev_n={r2['lev_norm']:.2f}  "
                  f"{r2['vi_i'][:50]!r}  <->  {r2['vi_j'][:50]!r}")
    else:
        print("\n  (no surviving collisions — AND-gate vetoed every v1 collision)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
