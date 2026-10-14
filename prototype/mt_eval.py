#!/usr/bin/env python3
"""Evaluate the MT stage of the cascade (vi -> en/zh/ko) on FLORES-200 devtest.

Computes BLEU + chrF2 with sacrebleu against line-aligned references.
This is the one pipeline stage that had never been quantitatively measured.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

# Force UTF-8 stdout (CJK output on Windows consoles otherwise dies).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import sacrebleu  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_fast import FastCascadeS2ST, FastPipelineConfig  # noqa: E402

PROJECT = HERE.parent
FLORES = PROJECT / "data" / "raw" / "flores200" / "flores200_dataset" / "devtest"

# pipeline target key -> FLORES devtest filename
FLORES_FILE = {
    "vi": "vie_Latn.devtest",
    "en": "eng_Latn.devtest",
    "zh": "zho_Hans.devtest",
    "ko": "kor_Hang.devtest",
}
# sacrebleu BLEU tokenizer per target ("13a" default; zh needs char tokenizer)
BLEU_TOK = {"en": "13a", "zh": "zh", "ko": "13a"}


def load_lines(lang: str) -> list[str]:
    p = FLORES / FLORES_FILE[lang]
    with open(p, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--targets", default="en,zh,ko",
                    help="comma list of pipeline target keys")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    src_all = load_lines("vi")
    refs_all = {t: load_lines(t) for t in targets}
    total = len(src_all)
    for t in targets:
        assert len(refs_all[t]) == total, f"misaligned {t}"

    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(total), min(args.n, total)))
    src = [src_all[i] for i in idx]
    print(f"[data] FLORES-200 devtest, total={total} lines, "
          f"sampled n={len(idx)} seed={args.seed}")
    print(f"[cfg]  mt_beam={args.beam}  targets={targets}")

    cfg = FastPipelineConfig()
    cfg.mt_beam = args.beam
    pipe = FastCascadeS2ST(cfg)

    results = {}
    for t in targets:
        ref = [refs_all[t][i] for i in idx]
        print(f"\n[run] vi -> {t}  ({len(src)} sentences, beam={args.beam}) ...",
              flush=True)
        t0 = time.perf_counter()
        hyp = pipe.translate_sentences_batch(src, t, source="vi")
        dt = time.perf_counter() - t0
        sps = dt / max(1, len(src))

        bleu = sacrebleu.corpus_bleu(hyp, [ref], tokenize=BLEU_TOK[t])
        chrf = sacrebleu.corpus_chrf(hyp, [ref])  # chrF2 (beta=2) by default
        results[t] = dict(bleu=bleu.score, chrf=chrf.score, dt=dt, sps=sps,
                          src=src, ref=ref, hyp=hyp)
        print(f"[done] vi->{t}  BLEU={bleu.score:.2f}  chrF2={chrf.score:.2f}  "
              f"time={dt:.1f}s  ({sps:.3f} s/sent)")

    # ---- summary table ----
    print("\n" + "=" * 64)
    print(f"MT EVAL  (FLORES-200 devtest, n={len(idx)}, seed={args.seed}, "
          f"beam={args.beam})")
    print("=" * 64)
    print(f"{'target':<8}{'BLEU':>8}{'chrF2':>9}{'s/sent':>10}{'BLEU_tok':>10}")
    print("-" * 64)
    for t in targets:
        r = results[t]
        print(f"vi->{t:<5}{r['bleu']:>8.2f}{r['chrf']:>9.2f}"
              f"{r['sps']:>10.3f}{BLEU_TOK[t]:>10}")
    print("=" * 64)

    # ---- example triples ----
    for t in targets:
        r = results[t]
        print(f"\n--- examples vi->{t} (3) ---")
        for k in range(min(3, len(src))):
            print(f"[SRC] {r['src'][k]}")
            print(f"[REF] {r['ref'][k]}")
            print(f"[HYP] {r['hyp'][k]}")
            print()


if __name__ == "__main__":
    main()
