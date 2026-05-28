"""
T2 diacritic-stripping collision audit (ACL reviewer Q6).

Reviewer concern: the T2 text cache strips Vietnamese diacritics, risking
conflation of semantically distinct forms. This script puts a real number
on how often that actually happens.

Protocol:
  1. Load every unique Vietnamese transcript from VIVOS train prompts.txt
     (with an industrial-phrase + tonal-minimal-pair fallback corpus).
  2. Compute the 256-d hashed n-gram fingerprint (translation_memory).
  3. For every unordered pair (s_i, s_j), flag a collision when
        cos_T2(s_i, s_j) >= tau_T2  AND
        Levenshtein(norm_i, norm_j) > 0.3 * max(len_i, len_j),
     where norm_* preserves diacritics (lower + strip punct + ws collapse).
     i.e. the fingerprint says "near-identical" but the surface forms are
     more than 30% different — a genuine semantic-conflation risk.
  4. Save Results/t2_collision_audit.json with totals + top-10 worst.
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

from translation_memory import text_fingerprint  # noqa: E402

PROJECT     = HERE.parent
VIVOS_TRAIN = PROJECT / "data" / "processed" / "vivos" / "vivos" / "train"

TAU_T2 = 0.80
LEV_FRAC = 0.30

_PUNCT_RE = re.compile(r"[^\w\s\xC0-ỹ]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def diacritic_preserving_normalize(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace, BUT keep diacritics.
    Mirrors translation_memory._normalize except we do NOT NFD-strip combining
    marks; this is the "true" surface form we compare Levenshtein on."""
    if not text:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def diacritic_stripped(text: str) -> str:
    """Matches translation_memory._normalize byte-for-byte."""
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
        """Fallback pure-Python Levenshtein — slow but correct."""
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
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


# ---------------- Corpus loading ----------------

def _load_vivos_unique_transcripts() -> list[str]:
    pf = VIVOS_TRAIN / "prompts.txt"
    if not pf.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in pf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, _, text = line.partition(" ")
        text = text.strip()
        if not text:
            continue
        # canonicalize case for dedup but keep original casing for display
        key = diacritic_preserving_normalize(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _load_fallback_corpus() -> list[str]:
    """Industrial phrases + curated tonal minimal-pair Vietnamese sentences.
    Used only if VIVOS prompts.txt is unavailable."""
    industrial: list[str] = []
    ip_file = HERE / "industrial_phrases.json"
    if ip_file.exists():
        ip = json.loads(ip_file.read_text(encoding="utf-8"))
        for k, v in ip.items():
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "vi" in item:
                        industrial.append(item["vi"])

    # Curated Vietnamese tonal minimal-pair sentences (50 items).
    minimal_pairs = [
        "tôi thấy con ma trên đường",       "tôi thấy con má trên đường",
        "anh ấy là ba của tôi",             "anh ấy là bà của tôi",
        "cô ấy nói la to",                  "cô ấy nói lá to",
        "đặt cái ca lên bàn",               "đặt cái cá lên bàn",
        "đứa trẻ đang khóc",                "đứa trẻ đang khoc",
        "ông ấy đang đi",                   "ông ấy đáng đi",
        "tôi muốn mua hai cái",             "tôi muốn mưa hai cái",
        "đèn đỏ phía trước",                "đèn đo phía trước",
        "máy đang chạy",                    "máy đáng chạy",
        "tôi đến từ hà nội",                "tôi đền từ hà nội",
        "anh có biết không",                "anh có biết khong",
        "cô ấy mặc áo dài",                 "cô ấy mac áo dài",
        "ngày mai trời mưa",                "ngày mai trời mua",
        "tôi đang học tiếng việt",          "tôi đáng học tiếng việt",
        "mẹ tôi nấu cơm",                   "mẹ tôi nâu cơm",
        "đường này rất dài",                "đường này rất dai",
        "tôi muốn uống nước",               "tôi muôn uống nước",
        "cảm ơn rất nhiều",                 "cam ơn rất nhiều",
        "xin chào các bạn",                 "xin chao các bạn",
        "hôm nay là thứ hai",               "hôm nay là thứ hái",
        "tôi yêu việt nam",                 "tôi yếu việt nam",
        "bạn có khỏe không",                "bạn có khoẻ khong",
        "đi thẳng phía trước",              "đi thang phía trước",
        "rẽ trái ở ngã tư",                 "rẽ trai ở ngã tư",
        "tôi cần giúp đỡ",                  "tôi cân giúp đỡ",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for t in industrial + minimal_pairs:
        key = diacritic_preserving_normalize(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ---------------- Audit ----------------

def audit(corpus: list[str], tau_t2: float = TAU_T2,
          lev_frac: float = LEV_FRAC, chunk: int = 512) -> dict:
    """Memory-safe: stream the upper triangle of the cosine matrix in
    chunk-of-rows × full-cols slabs of float32. Never materialize the full
    n×n matrix (which is 4 GB at n=10K)."""
    n = len(corpus)
    print(f"[fp] computing {n} fingerprints ...", flush=True)
    t0 = time.perf_counter()
    fps = np.stack([text_fingerprint(t) for t in corpus], axis=0).astype(np.float32)
    norm_orig = [diacritic_preserving_normalize(t) for t in corpus]
    print(f"  done in {time.perf_counter()-t0:.2f}s "
          f"shape={fps.shape} mem={fps.nbytes/1e6:.1f}MB",
          flush=True)

    total_pairs = n * (n - 1) // 2
    print(f"[scan] streaming {total_pairs} pair cosines in row-chunks of "
          f"{chunk} (no n×n materialization) ...", flush=True)

    # Precompute lengths for fast pre-pruning.
    lens = np.asarray([len(s) for s in norm_orig], dtype=np.int32)

    collisions: list[dict] = []
    n_above = 0
    n_autocoll = 0   # length-difference auto-collisions (no Lev needed)
    n_lev_calls = 0
    t0 = time.perf_counter()
    for r0 in range(0, n, chunk):
        r1 = min(r0 + chunk, n)
        block = fps[r0:r1] @ fps.T              # (chunk, n) float32
        for li, i in enumerate(range(r0, r1)):
            row = block[li]
            j_idx = np.flatnonzero(row >= tau_t2)
            j_idx = j_idx[j_idx > i]
            n_above += int(j_idx.size)
            if j_idx.size == 0:
                continue
            a = norm_orig[i]
            la = lens[i]
            if not a:
                continue
            # Vectorize the "length-only auto-collision" check:
            #   if abs(la - lb) > lev_frac * max(la, lb), Lev >= abs diff > thresh
            #   → collision is guaranteed; skip Lev.
            jb = j_idx
            lb = lens[jb]
            maxL = np.maximum(la, lb)
            absdiff = np.abs(la - lb)
            auto_mask = absdiff > (lev_frac * maxL)
            # remaining candidates need real Lev
            need_lev_idx = jb[~auto_mask]
            auto_idx     = jb[auto_mask]

            for j in auto_idx.tolist():
                b = norm_orig[j]
                if not b:
                    continue
                L = int(max(la, lens[j]))
                d = int(abs(la - lens[j]))
                collisions.append({
                    "i": int(i), "j": int(j),
                    "vi_i": corpus[i], "vi_j": corpus[j],
                    "stripped_i": diacritic_stripped(corpus[i]),
                    "stripped_j": diacritic_stripped(corpus[j]),
                    "cos_t2": float(row[j]),
                    "lev": d, "max_len": L,
                    "lev_norm": float(d / L) if L else 0.0,
                    "via": "length_diff",
                })
                n_autocoll += 1

            for j in need_lev_idx.tolist():
                b = norm_orig[j]
                if not b:
                    continue
                L = int(max(la, lens[j]))
                d = levenshtein(a, b)
                n_lev_calls += 1
                if d > lev_frac * L:
                    collisions.append({
                        "i": int(i), "j": int(j),
                        "vi_i": corpus[i], "vi_j": corpus[j],
                        "stripped_i": diacritic_stripped(corpus[i]),
                        "stripped_j": diacritic_stripped(corpus[j]),
                        "cos_t2": float(row[j]),
                        "lev": int(d), "max_len": L,
                        "lev_norm": float(d / L) if L else 0.0,
                        "via": "lev",
                    })
        if r0 % (chunk * 4) == 0 or r1 == n:
            elapsed = time.perf_counter() - t0
            done = r1 / n
            eta = elapsed / max(done, 1e-6) - elapsed
            print(f"  row {r1}/{n} ({done*100:.1f}%) "
                  f"above_tau={n_above} auto_coll={n_autocoll} "
                  f"lev_calls={n_lev_calls} total_coll={len(collisions)} "
                  f"elapsed={elapsed:.1f}s eta={eta:.0f}s", flush=True)

    print(f"  scan done in {time.perf_counter()-t0:.1f}s", flush=True)
    collisions.sort(key=lambda r: (-r["cos_t2"], -r["lev_norm"]))
    n_coll = len(collisions)
    rate = n_coll / total_pairs if total_pairs else 0.0

    print(f"  collisions: {n_coll} / {total_pairs} pairs "
          f"({rate*100:.4f}%)", flush=True)

    return {
        "config": {
            "tau_t2": tau_t2, "lev_frac": lev_frac,
            "n_corpus": n, "total_pairs": total_pairs,
            "n_above_tau": int(n_above),
            "n_collisions": n_coll,
            "collision_rate": rate,
        },
        "worst_10": collisions[:10],
        "all_collisions": collisions,
    }


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=TAU_T2)
    ap.add_argument("--lev-frac", type=float, default=LEV_FRAC)
    ap.add_argument("--max-corpus", type=int, default=3000,
                    help="cap corpus size (0 = no cap; default 3000 makes the "
                    "scan tractable while remaining statistically representative)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="Results/t2_collision_audit.json")
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("[corpus] loading VIVOS train prompts ...", flush=True)
    corpus = _load_vivos_unique_transcripts()
    source = "vivos_train_prompts"
    if not corpus:
        print("  VIVOS unavailable, falling back to industrial + minimal-pair set",
              flush=True)
        corpus = _load_fallback_corpus()
        source = "fallback_industrial_minpair"
    print(f"  loaded {len(corpus)} unique transcripts (source={source})",
          flush=True)

    if args.max_corpus and len(corpus) > args.max_corpus:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(corpus), args.max_corpus, replace=False)
        corpus = [corpus[i] for i in sorted(idx)]
        print(f"  sub-sampled to {len(corpus)} (seed={args.seed})", flush=True)

    result = audit(corpus, tau_t2=args.tau, lev_frac=args.lev_frac)
    result["source"] = source

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[done] wrote {out_path}")

    print("\n" + "=" * 78)
    print(f"T2 COLLISION AUDIT — tau>={args.tau}, lev>={args.lev_frac}*max_len")
    print("=" * 78)
    c = result["config"]
    print(f"  corpus size                : {c['n_corpus']:>8}")
    print(f"  total unordered pairs      : {c['total_pairs']:>8}")
    print(f"  pairs above cosine tau     : {c['n_above_tau']:>8}")
    print(f"  collision pairs            : {c['n_collisions']:>8}")
    print(f"  collision rate             : {c['collision_rate']*100:>8.4f}%")

    if result["worst_10"]:
        print("\n  worst-10 collisions (cos, lev_norm, vi_i  <->  vi_j):")
        for r in result["worst_10"]:
            print(f"    cos={r['cos_t2']:.3f} lev_n={r['lev_norm']:.2f}  "
                  f"{r['vi_i'][:55]!r}  <->  {r['vi_j'][:55]!r}")
    else:
        print("\n  (no collisions detected — diacritic-stripping is safe at "
              "this tau)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
