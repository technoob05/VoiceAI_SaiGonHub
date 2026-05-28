"""
Adversarial Vietnamese minimal-pair SENTENCE evaluation (reviewer Q on
syllable-level vs sentence-level confusability).

Reviewer concern: the 30-clip syllable-level minimal-pair corpus of
sec:exp:twostream uses a single carrier-syllable substitution; in deployment
the tonal contrast lives inside a longer sentence whose surrounding context
dilutes the tone signal at the L2-cosine level. We test that hypothesis
directly by hand-curating 30 Vietnamese minimal-pair SENTENCES from common
homophone confusions (data/adversarial_minimal_pairs_vi.json), synthesizing
both members via MMS-TTS-vie at seed 42, inserting source_a into the two-stream
cache, then probing with source_b. Report the false-hit rate at the deployment
default (tau_A=0.85, tau_B=0.93) and at the Round-25 auto-calibrated point.

Outputs:
  Results/adversarial_minimal_pair_eval.json

Run from project root:
    python prototype/adversarial_minimal_pair_eval.py
"""
from __future__ import annotations

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

import numpy as np                                              # noqa: E402

from two_stream_voting import (                                  # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
)

ADV_JSON = PROJECT / "data" / "adversarial_minimal_pairs_vi.json"

# Operating points to probe: deployment default + Round-25 auto-calibrated
# point selected by the precision-floored grid recipe of sec:exp:pr.
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


def _existence_check(fp_query, fp_cached, tau_A, tau_B):
    t = float(np.dot(fp_query.tone, fp_cached.tone))
    s = float(np.dot(fp_query.segment, fp_cached.segment))
    return (t >= tau_A and s >= tau_B), t, s


def main() -> int:
    print("[load] adversarial pairs ...")
    payload = json.loads(ADV_JSON.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    print(f"  {len(pairs)} sentence pairs")

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    print(f"[synth] generating {2 * len(pairs)} clips at seed {SEED} ...")
    fp_helper = TwoStreamCache()
    pair_records = []
    for i, p in enumerate(pairs):
        a_wav = _synth(p["source_a"], tok, model)
        b_wav = _synth(p["source_b"], tok, model)
        fp_a = fp_helper.fingerprint(a_wav, sr)
        fp_b = fp_helper.fingerprint(b_wav, sr)
        # Diagnostic raw cosines (no threshold yet)
        t_cos = float(np.dot(fp_a.tone,    fp_b.tone))
        s_cos = float(np.dot(fp_a.segment, fp_b.segment))
        pair_records.append({
            "idx":       i,
            "source_a":  p["source_a"],
            "source_b":  p["source_b"],
            "gloss_a":   p["gloss_en_a"],
            "gloss_b":   p["gloss_en_b"],
            "fp_a":      fp_a,
            "fp_b":      fp_b,
            "tone_cos":     round(t_cos, 4),
            "segment_cos":  round(s_cos, 4),
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(pairs)}")

    # Per-pair fingerprint diagnostic table (does cos(fp_a,fp_b) clear thresholds?)
    print(f"\n[diag] mean tone_cos={np.mean([r['tone_cos'] for r in pair_records]):.4f}, "
          f"mean segment_cos={np.mean([r['segment_cos'] for r in pair_records]):.4f}")

    # ---- Evaluate at each operating point ----
    results_by_point = {}
    print("\n[eval] existence-rule false-hit at deployment thresholds ...")
    for name, tau_A, tau_B in OP_POINTS:
        false_hits = 0
        per_pair = []
        for r in pair_records:
            cache = TwoStreamCache(cfg=TwoStreamConfig(
                tone_threshold=tau_A, segment_threshold=tau_B))
            cache.put(r["fp_a"], "en", {
                "vi":  r["source_a"],
                "tgt": r["gloss_a"],
            })
            hit = cache.lookup_existence(r["fp_b"], "en")
            is_false_hit = hit is not None
            if is_false_hit:
                false_hits += 1
            per_pair.append({
                "idx":          r["idx"],
                "tone_cos":     r["tone_cos"],
                "segment_cos":  r["segment_cos"],
                "false_hit":    bool(is_false_hit),
            })
        false_hit_rate = false_hits / len(pair_records)
        results_by_point[name] = {
            "tau_A":          tau_A,
            "tau_B":          tau_B,
            "n_pairs":        len(pair_records),
            "false_hits":     false_hits,
            "false_hit_rate": round(false_hit_rate, 4),
            "per_pair":       per_pair,
        }
        print(f"  ({name}) tau_A={tau_A} tau_B={tau_B}: "
              f"false-hits = {false_hits}/{len(pair_records)} "
              f"({false_hit_rate*100:.1f}%)")

    # ---- Comparison vs syllable-level baseline ----
    # syllable-level @ deployment default: 7/25 = 28% false-hit (Results/two_stream_existence_eval.json)
    syllable_false_hit_default = 7 / 25
    delta = round(results_by_point["deployment_default"]["false_hit_rate"]
                  - syllable_false_hit_default, 4)

    comparison_table = {
        "syllable_level_30clip_default":  round(syllable_false_hit_default, 4),
        "sentence_level_adversarial_default":
            results_by_point["deployment_default"]["false_hit_rate"],
        "delta_sentence_minus_syllable":  delta,
        "interpretation":
            "Positive delta means sentence-level adversarial probes are HARDER for the cache "
            "(longer context dilutes the tonal contrast at L2 cosine), confirming the reviewer hypothesis. "
            "Negative delta means the adversarial set is easier than expected.",
    }
    print(f"\n[compare] syllable false-hit = {comparison_table['syllable_level_30clip_default']}, "
          f"sentence-level adversarial false-hit = {comparison_table['sentence_level_adversarial_default']}, "
          f"delta = {delta:+.4f}")

    out = {
        "seed":         SEED,
        "n_pairs":      len(pair_records),
        "source":       "data/adversarial_minimal_pairs_vi.json (author-curated)",
        "tts_model":    "facebook/mms-tts-vie",
        "tts_sr":       sr,
        "operating_points": [
            {"name": n, "tau_A": a, "tau_B": b} for n, a, b in OP_POINTS
        ],
        "results":           {k: {kk: vv for kk, vv in v.items() if kk != "per_pair"}
                              for k, v in results_by_point.items()},
        "per_pair_default":  results_by_point["deployment_default"]["per_pair"],
        "comparison_vs_syllable_level": comparison_table,
        "headline":
            f"At (tau_A=0.85, tau_B=0.93): {results_by_point['deployment_default']['false_hits']}/30 "
            f"adversarial sentence-level minimal pairs false-hit "
            f"({results_by_point['deployment_default']['false_hit_rate']*100:.1f}%). "
            f"At auto-calibrated (0.70, 0.92): "
            f"{results_by_point['auto_calibrated']['false_hits']}/30 "
            f"({results_by_point['auto_calibrated']['false_hit_rate']*100:.1f}%). "
            f"Delta vs syllable-level baseline: {delta:+.4f}.",
    }

    out_path = PROJECT / "Results" / "adversarial_minimal_pair_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    print(f"[headline] {out['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
