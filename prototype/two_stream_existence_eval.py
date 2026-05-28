"""
Existence-rule vs argmax-equality rule for Algorithm 2 (reviewer Q1/W1).

The original rule (Algorithm 2 in main.tex) requires both streams to nominate
the SAME stored entry as argmax. A reviewer pointed out this is unnecessarily
restrictive: an entry e may satisfy both thresholds yet not be the top
candidate for one of the streams, and would be needlessly vetoed.

This script runs both rules on the same 30-clip minimal-pair corpus
(5 syllables x 6 tones) and reports self-lookup recall, correct-miss
rate, and false-hit rate under each rule.

Run from project root:
    python prototype/two_stream_existence_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

import numpy as np                                            # noqa: E402
from two_stream_voting import (                                # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
    MINIMAL_PAIR_SYLLABLES,
    CARRIER,
    _synth,
)


def _eval_rule(audio_by_key, sr, rule_name):
    """Run self-lookup (exp1) + tonal-variant lookup (exp2) under one rule.

    rule_name is "argmax_equality" (original) or "existence" (reviewer fix).
    """
    def _lookup(cache, fp):
        if rule_name == "argmax_equality":
            return cache.lookup(fp, "en")
        if rule_name == "existence":
            return cache.lookup_existence(fp, "en")
        raise ValueError(rule_name)

    # exp1: store all variants, query each — should self-hit
    cache_full = TwoStreamCache()
    for k, a in audio_by_key.items():
        fp = cache_full.fingerprint(a, sr)
        cache_full.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})

    self_hits = 0
    for k, a in audio_by_key.items():
        fp = cache_full.fingerprint(a, sr)
        h = _lookup(cache_full, fp)
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1

    # exp2: store ONLY ngang of each syllable, query OTHER tones — should MISS
    cache_ngang = TwoStreamCache()
    for base in MINIMAL_PAIR_SYLLABLES:
        ma_key = (base, MINIMAL_PAIR_SYLLABLES[base][0])
        if ma_key in audio_by_key:
            fp = cache_ngang.fingerprint(audio_by_key[ma_key], sr)
            cache_ngang.put(fp, "en", {
                "vi":  MINIMAL_PAIR_SYLLABLES[base][0],
                "tgt": MINIMAL_PAIR_SYLLABLES[base][0],
            })

    false_hits = 0
    correct_misses = 0
    total = 0
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audio_by_key:
                continue
            fp = cache_ngang.fingerprint(audio_by_key[k], sr)
            h = _lookup(cache_ngang, fp)
            total += 1
            if h is None:
                correct_misses += 1
            else:
                false_hits += 1

    return {
        "rule":                            rule_name,
        "self_lookups":                    len(audio_by_key),
        "self_hits":                       self_hits,
        "self_hit_rate":                   round(self_hits / max(len(audio_by_key), 1), 4),
        "tonal_variant_lookups":           total,
        "tonal_variant_correct_misses":    correct_misses,
        "tonal_variant_correct_miss_rate": round(correct_misses / max(total, 1), 4),
        "tonal_variant_false_hits":        false_hits,
        "tonal_variant_false_hit_rate":    round(false_hits / max(total, 1), 4),
        "cache_stats_exp2":                cache_ngang.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Results/two_stream_existence_eval.json")
    args = ap.parse_args()

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    audio_by_key = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] {v}: {e}")
    print(f"[synth] {len(audio_by_key)} clips")

    print("\n[rule] argmax_equality (original Algorithm 2) ...")
    out_argmax = _eval_rule(audio_by_key, sr, "argmax_equality")
    print(f"  self-lookup           : {out_argmax['self_hits']}/{out_argmax['self_lookups']}")
    print(f"  correct-miss (tonal)  : {out_argmax['tonal_variant_correct_misses']}/{out_argmax['tonal_variant_lookups']}")
    print(f"  false-hit (tonal)     : {out_argmax['tonal_variant_false_hits']}/{out_argmax['tonal_variant_lookups']}")

    print("\n[rule] existence (reviewer-fixed Algorithm 2) ...")
    out_exist = _eval_rule(audio_by_key, sr, "existence")
    print(f"  self-lookup           : {out_exist['self_hits']}/{out_exist['self_lookups']}")
    print(f"  correct-miss (tonal)  : {out_exist['tonal_variant_correct_misses']}/{out_exist['tonal_variant_lookups']}")
    print(f"  false-hit (tonal)     : {out_exist['tonal_variant_false_hits']}/{out_exist['tonal_variant_lookups']}")

    diff = {
        "self_hit_rate_delta":              round(out_exist["self_hit_rate"]              - out_argmax["self_hit_rate"], 4),
        "correct_miss_rate_delta":          round(out_exist["tonal_variant_correct_miss_rate"] - out_argmax["tonal_variant_correct_miss_rate"], 4),
        "false_hit_rate_delta":             round(out_exist["tonal_variant_false_hit_rate"] - out_argmax["tonal_variant_false_hit_rate"], 4),
    }
    print("\n[diff] existence - argmax_equality:")
    for k, v in diff.items():
        print(f"  {k:32s} {v:+.4f}")

    out = {
        "argmax_equality_rule": out_argmax,
        "existence_rule":       out_exist,
        "diff_existence_minus_argmax": diff,
        "tonal_protection_100pct_under_existence":
            out_exist["tonal_variant_correct_miss_rate"] == 1.0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
