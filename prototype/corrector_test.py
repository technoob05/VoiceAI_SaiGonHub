"""corrector_test.py — does a Vietnamese error-correction model actually help our ASR?

Loads (ref, hyp) pairs saved by asr_eval.py, runs a Vietnamese correction model on
each ASR hypothesis, and recomputes WER/CER. If corrected-WER < raw-WER the
corrector helps; if ≥, it doesn't (our errors are acoustic word-substitutions, not
spelling/diacritics — a corrector can't recover a word the recognizer never heard,
and may introduce new errors).

    python corrector_test.py --pairs asr_eval_pairs_beam1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from asr_eval import norm, wer, cer, strip_diacritics  # reuse metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="asr_eval_pairs_beam1.json")
    ap.add_argument("--model", default="bmd1905/vietnamese-correction-v2")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    pairs = json.load(open(HERE / args.pairs, encoding="utf-8"))
    print(f"[corrector] {len(pairs)} pairs from {args.pairs}")
    print(f"[corrector] loading {args.model} …")
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()

    def corr(text: str) -> str:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            g = model.generate(**enc, max_length=256, num_beams=1)
        return tok.decode(g[0], skip_special_tokens=True)

    W_raw = W_cor = Wt = 0
    C_raw = C_cor = Ct = 0
    changed = helped = hurt = 0
    examples = []
    for p in pairs:
        ref, hyp = p["ref"], p["hyp"]
        cor = norm(corr(hyp))
        we_r, wt = wer(ref, hyp); we_c, _ = wer(ref, cor)
        ce_r, ct = cer(ref, hyp); ce_c, _ = cer(ref, cor)
        W_raw += we_r; W_cor += we_c; Wt += wt
        C_raw += ce_r; C_cor += ce_c; Ct += ct
        if cor != hyp:
            changed += 1
            if we_c < we_r: helped += 1
            elif we_c > we_r: hurt += 1
            if len(examples) < args.show:
                examples.append((ref, hyp, cor, we_r, we_c))

    print("\n" + "=" * 70)
    print(f"  RAW  ASR : WER = {W_raw/max(Wt,1)*100:.2f}%   CER = {C_raw/max(Ct,1)*100:.2f}%")
    print(f"  +CORRECTOR: WER = {W_cor/max(Wt,1)*100:.2f}%   CER = {C_cor/max(Ct,1)*100:.2f}%")
    delta = (W_cor - W_raw) / max(Wt, 1) * 100
    verdict = "HELPS" if delta < -0.3 else ("HURTS" if delta > 0.3 else "NO EFFECT")
    print(f"  Δ WER = {delta:+.2f} pp  → corrector {verdict}")
    print(f"  changed {changed}/{len(pairs)} lines  (helped {helped}, hurt {hurt})")
    print("=" * 70)
    print("\nExamples (ref / hyp / corrected):")
    for ref, hyp, cor, wr, wc in examples:
        flag = "↓better" if wc < wr else ("↑worse" if wc > wr else "=same")
        print(f"  [{flag}]")
        print(f"    ref: {ref}")
        print(f"    hyp: {hyp}")
        print(f"    cor: {cor}")


if __name__ == "__main__":
    main()
