"""
Tonal minimal-pair eval — LARGE corpus (20 syllables x 6 tones = 120 clips).

Response to ACL reviewer's repeated concern that the original eval (30-36
clips, clean TTS only) is too small to support the §4.3-§4.4 claims. We
expand on two axes:

  (a) Carrier inventory: 6 -> 20 Vietnamese carrier syllables.
  (b) Acoustic conditions: clean TTS + 4 additive-noise SNR conditions
      (20 / 15 / 10 / 5 dB) to test whether existence-rule two-stream
      voting holds up under industrial-noise stress.

Pipeline (CPU-only, deterministic):
  1. For each of 20 syllables x 6 tones, synthesize "noi {syl}" via
     facebook/mms-tts-vie (deterministic forward pass).
  2. For each condition in {clean, 20dB, 15dB, 10dB, 5dB}, build a
     TwoStreamCache, store the ngang variant of each syllable, then probe
     the other 5 tonal variants AND the ngang variants themselves.
  3. Run existence-rule lookups at (tau_A, tau_B) in {(0.85, 0.93),
     (0.90, 0.93)}. Report self-lookup recall, correct-miss rate on tonal
     variants, false-hit rate.

Noise model: since DEMAND/MUSAN are not on the local development host, we
synthesize a DEMAND-like broadband background = pink-tilted Gaussian +
band-limited low-frequency rumble, scaled to the requested SNR per clip
(same convention used by `noise_robustness_eval.py` in this prototype).
This is documented as a limitation in the manuscript.

Output: Results/minimal_pair_eval_large.json
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

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                            # noqa: E402

from two_stream_voting import (                                # noqa: E402
    TwoStreamCache,
    TwoStreamConfig,
)


CARRIER = "noi {syl}"

# 20-syllable corpus -- each syllable carries the 6 Vietnamese tones.
# Tones: ngang (level), sac (high-rising), hoi (dipping), nang (low-glottal),
#         nga (creaky-rising), huyen (low-falling).
MINIMAL_PAIR_SYLLABLES_20 = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
    "ta":  ["ta",  "tá",  "tả",  "tạ",  "tã",  "tà"],
    "na":  ["na",  "ná",  "nả",  "nạ",  "nã",  "nà"],
    "sa":  ["sa",  "sá",  "sả",  "sạ",  "sã",  "sà"],
    "ha":  ["ha",  "há",  "hả",  "hạ",  "hã",  "hà"],
    "va":  ["va",  "vá",  "vả",  "vạ",  "vã",  "và"],
    "pa":  ["pa",  "pá",  "pả",  "pạ",  "pã",  "pà"],
    "ga":  ["ga",  "gá",  "gả",  "gạ",  "gã",  "gà"],
    "ka":  ["ka",  "ká",  "kả",  "kạ",  "kã",  "kà"],
    "ra":  ["ra",  "rá",  "rả",  "rạ",  "rã",  "rà"],
    "xa":  ["xa",  "xá",  "xả",  "xạ",  "xã",  "xà"],
    "gia": ["gia", "giá", "giả", "giạ", "giã", "già"],
    "kha": ["kha", "khá", "khả", "khạ", "khã", "khà"],
    "nga": ["nga", "ngá", "ngả", "ngạ", "ngã", "ngà"],
    "nha": ["nha", "nhá", "nhả", "nhạ", "nhã", "nhà"],
    "tha": ["tha", "thá", "thả", "thạ", "thã", "thà"],
}


def _seed_all(seed: int = 20260528) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def synth_audio(text: str, tok, model, sr: int) -> np.ndarray:
    import torch
    inputs = tok(text, return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform
    return wav.squeeze().cpu().numpy().astype(np.float32)


def _demand_like_noise(n: int, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    """Synthetic DEMAND-like broadband background.

    DEMAND PSTATION / DKITCHEN are dominated by low-frequency rumble + broadband
    hiss (HVAC, machinery, distant chatter). We approximate this as pink-tilted
    noise plus a low-band rumble component. Unit-RMS output; caller scales to
    the requested SNR.
    """
    white = rng.standard_normal(n).astype(np.float32)
    # Pink-ish tilt via 1-pole low-pass + scaling
    pink = np.empty_like(white)
    pink[0] = white[0]
    a = 0.92
    for i in range(1, n):
        pink[i] = a * pink[i-1] + (1 - a) * white[i]
    # Add low-band rumble (~80-200 Hz)
    t = np.arange(n) / sr
    rumble = (rng.standard_normal(n).astype(np.float32) * 0.0
              + np.sin(2 * np.pi * 110 * t + rng.uniform(0, 2*np.pi)).astype(np.float32) * 0.4
              + np.sin(2 * np.pi *  85 * t + rng.uniform(0, 2*np.pi)).astype(np.float32) * 0.3)
    noise = 0.7 * pink + 0.3 * rumble
    rms = float(np.sqrt((noise ** 2).mean() + 1e-12))
    return (noise / max(rms, 1e-9)).astype(np.float32)


def _mix_at_snr(audio: np.ndarray, snr_db: float, rng: np.random.Generator, sr: int) -> np.ndarray:
    sig_power = float((audio ** 2).mean() + 1e-12)
    noise_unit = _demand_like_noise(len(audio), rng, sr=sr)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise_unit * np.sqrt(target_noise_power)
    return (audio + noise).astype(np.float32)


def _build_corpus(tok, model, sr: int):
    """Synthesize every (base, variant) pair. Deterministic."""
    audio_by_key: dict[tuple, np.ndarray] = {}
    n_total = sum(len(v) for v in MINIMAL_PAIR_SYLLABLES_20.values())
    i = 0
    t0 = time.perf_counter()
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        for v in variants:
            i += 1
            text = CARRIER.format(syl=v)
            try:
                a = synth_audio(text, tok, model, sr)
                audio_by_key[(base, v)] = a
            except Exception as e:
                print(f"  [warn] synth failed for {text!r}: {e}")
            if i % 20 == 0:
                print(f"  [synth] {i}/{n_total}  elapsed={time.perf_counter()-t0:5.1f}s")
    print(f"[synth] generated {len(audio_by_key)}/{n_total} clips "
          f"in {time.perf_counter()-t0:.1f}s")
    return audio_by_key


def _build_condition_audios(clean_audio_by_key: dict, snr_db: float | None, sr: int):
    """Return condition-specific dict of clip key -> (possibly-noisy) audio.

    Use a deterministic per-clip noise seed so a re-run is reproducible.
    """
    if snr_db is None:
        return clean_audio_by_key
    out = {}
    for idx, (k, a) in enumerate(clean_audio_by_key.items()):
        rng = np.random.default_rng(20260528 + int(snr_db * 100) + idx)
        out[k] = _mix_at_snr(a, snr_db, rng, sr)
    return out


def _run_eval_one_condition(audios_by_key: dict, sr: int,
                            tau_A: float, tau_B: float) -> dict:
    """Run the existence-rule two-stream eval at a given operating point.

    Protocol:
      - Cache stores the ngang (level-tone) variant of each base syllable.
      - Self-lookup: query each stored entry against itself (target: hit).
      - Tonal-variant probe: query the other 5 tones per base syllable;
        each should MISS (the cache only knows ngang). A hit is a false hit.
      - Also: query every NON-stored clip as a self-lookup recall reference
        across all 120 clips (store-then-lookup-itself).
    """
    cfg = TwoStreamConfig(tone_threshold=tau_A, segment_threshold=tau_B, max_entries=1024)

    # Self-lookup recall: store EVERY clip, then query each clip against itself.
    cache_self = TwoStreamCache(cfg)
    fps_self = {}
    for k, a in audios_by_key.items():
        fp = cache_self.fingerprint(a, sr)
        fps_self[k] = fp
        cache_self.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})
    self_hits = 0
    for k, fp in fps_self.items():
        h = cache_self.lookup_existence(fp, "en")
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1
    n_clips = len(audios_by_key)
    self_recall = self_hits / max(n_clips, 1)

    # Tonal-variant test: store only ngang per base, probe the other 5 tones.
    cache_t = TwoStreamCache(cfg)
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        ngang_key = (base, variants[0])
        if ngang_key in audios_by_key:
            fp = cache_t.fingerprint(audios_by_key[ngang_key], sr)
            cache_t.put(fp, "en", {"vi": variants[0], "tgt": variants[0]})

    correct_misses = 0
    false_hits = 0
    total_probes = 0
    for base, variants in MINIMAL_PAIR_SYLLABLES_20.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audios_by_key:
                continue
            fp = cache_t.fingerprint(audios_by_key[k], sr)
            h = cache_t.lookup_existence(fp, "en")
            total_probes += 1
            if h is None:
                correct_misses += 1
            else:
                false_hits += 1

    return {
        "tau_A": tau_A,
        "tau_B": tau_B,
        "n_clips_total": n_clips,
        "self_hits": self_hits,
        "self_recall": round(self_recall, 4),
        "n_tonal_probes": total_probes,
        "correct_misses": correct_misses,
        "correct_miss_rate": round(correct_misses / max(total_probes, 1), 4),
        "false_hits": false_hits,
        "false_hit_rate": round(false_hits / max(total_probes, 1), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      default="Results/minimal_pair_eval_large.json")
    ap.add_argument("--snrs",     nargs="+", type=float, default=[20, 15, 10, 5],
                    help="Noisy conditions in dB. Empty list -> clean only.")
    ap.add_argument("--reduce-snr-budget", action="store_true",
                    help="If set, use only {20, 15, 10} dB to stay under 20 min CPU.")
    args = ap.parse_args()

    if args.reduce_snr_budget:
        args.snrs = [20, 15, 10]

    _seed_all()

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate
    print(f"[load] ready, sr={sr}, n_syllables=20, n_tones=6 -> {20*6}=120 clips target")

    # ---- Synthesize the 120 clean clips once ----
    audio_clean = _build_corpus(tok, model, sr)
    if len(audio_clean) < 120:
        print(f"[warn] only {len(audio_clean)} clips synthesized (target 120)")

    # ---- Build conditions ----
    conditions = [("clean", None)] + [(f"snr_{int(s)}dB", float(s)) for s in args.snrs]
    print(f"\n[conditions] {[c[0] for c in conditions]}")

    operating_points = [(0.85, 0.93), (0.90, 0.93)]

    all_results = {}
    for cname, snr_db in conditions:
        print(f"\n[cond] {cname}  snr_db={snr_db}")
        audios = _build_condition_audios(audio_clean, snr_db, sr)
        per_op = {}
        for (tA, tB) in operating_points:
            t0 = time.perf_counter()
            r = _run_eval_one_condition(audios, sr, tA, tB)
            print(f"   tau=({tA},{tB})  self={r['self_hits']}/{r['n_clips_total']}  "
                  f"correct-miss={r['correct_misses']}/{r['n_tonal_probes']} "
                  f"({r['correct_miss_rate']*100:5.1f}%)  "
                  f"false-hit={r['false_hits']}/{r['n_tonal_probes']} "
                  f"({r['false_hit_rate']*100:5.1f}%)  "
                  f"elapsed={time.perf_counter()-t0:4.1f}s")
            per_op[f"tau_A={tA}_tau_B={tB}"] = r
        all_results[cname] = {"snr_db": snr_db, "by_operating_point": per_op}

    out = {
        "n_syllables": 20,
        "n_tones": 6,
        "n_clips_target": 120,
        "n_clips_synth": len(audio_clean),
        "carrier_template": CARRIER,
        "syllable_groups": MINIMAL_PAIR_SYLLABLES_20,
        "snr_conditions_db": args.snrs,
        "operating_points": [{"tau_A": a, "tau_B": b} for (a, b) in operating_points],
        "noise_model": "synthetic DEMAND-like (pink-tilted noise + 85/110 Hz rumble); RMS-scaled to SNR per clip",
        "rule": "existence (cos>=tau_A AND cos>=tau_B for at least one cached entry)",
        "results_by_condition": all_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
