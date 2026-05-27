"""
Noise-robustness eval — does cache hit through noisy versions of the same phrase?

Industrial scenarios add varying ambient noise to repeated phrases. If the
fingerprint is too sensitive, a worker repeating the same phrase under
different noise conditions misses the cache → wasted cascade work.

Test: for each VIVOS test utterance, inject increasing white-noise SNR
and measure cosine to the clean original.

Pass criterion: at moderate noise (SNR > 10 dB), cosine should stay above
the cache hit threshold (0.95) so cache continues to fire.

This validates the "perceptual cache" claim — small acoustic perturbation
should not break the cache key.

Also tests whether the noise_filter pre-processor (spectral subtraction)
improves cache robustness: filtered noisy audio → fingerprint should match
the clean fingerprint better than unfiltered noisy.

Output: Results/noise_robustness_eval.json with per-SNR cosine curves.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                            # noqa: E402
import soundfile as sf                                         # noqa: E402

from utterance_cache import UtteranceCache                     # noqa: E402
from noise_filter import NoiseFilter                           # noqa: E402


PROJECT = HERE.parent
VIVOS_TEST = PROJECT / "data" / "processed" / "vivos" / "vivos" / "test"


def _gather_vivos_test(n: int = 10) -> list[Path]:
    waves = VIVOS_TEST / "waves"
    out = []
    if not waves.exists():
        return out
    for spk in sorted(waves.iterdir())[:5]:
        if not spk.is_dir(): continue
        for w in sorted(spk.glob("*.wav"))[:2]:
            out.append(w)
            if len(out) >= n: return out
    return out


def _snr_to_noise_amp(audio: np.ndarray, target_snr_db: float, rng) -> np.ndarray:
    sig_power = float((audio ** 2).mean())
    noise_power = sig_power / (10 ** (target_snr_db / 10))
    noise_std = np.sqrt(max(noise_power, 1e-12))
    return rng.normal(0, noise_std, len(audio)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=10, help="Number of VIVOS test utterances")
    ap.add_argument("--snrs",  nargs="+", type=float, default=[40, 30, 20, 15, 10, 5, 0, -5])
    ap.add_argument("--out",   default="Results/noise_robustness_eval.json")
    args = ap.parse_args()

    utts = _gather_vivos_test(args.n)
    print(f"[data] {len(utts)} VIVOS test utterances")

    ucache = UtteranceCache()
    nf = NoiseFilter()
    rng = np.random.default_rng(42)

    snr_curves = []
    for snr_db in args.snrs:
        cosines_raw = []
        cosines_filtered = []
        snr_actual_in  = []
        snr_actual_out = []
        for wav in utts:
            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1: audio = audio.mean(axis=1)

            # Reference fingerprint (clean)
            fp_clean = ucache.fingerprint(audio, sr).vec

            # Add noise at target SNR
            noise = _snr_to_noise_amp(audio, snr_db, rng)
            noisy = audio + noise
            fp_noisy = ucache.fingerprint(noisy, sr).vec

            # Filter + fingerprint
            nf_res = nf(noisy, sr)
            fp_filtered = ucache.fingerprint(nf_res.filtered_audio, nf_res.sr).vec

            if len(fp_clean) == len(fp_noisy):
                cosines_raw.append(float(np.dot(fp_clean, fp_noisy)))
            if len(fp_clean) == len(fp_filtered):
                cosines_filtered.append(float(np.dot(fp_clean, fp_filtered)))
            snr_actual_in.append(nf_res.snr_in_db_est)
            snr_actual_out.append(nf_res.snr_out_db_est)

        entry = {
            "snr_db_target":   float(snr_db),
            "snr_db_est_in_mean":  round(float(np.mean(snr_actual_in)), 2),
            "snr_db_est_out_mean": round(float(np.mean(snr_actual_out)), 2),
            "cache_cosine_raw_mean":       round(float(np.mean(cosines_raw)), 4),
            "cache_cosine_raw_std":        round(float(np.std(cosines_raw)),  4),
            "cache_cosine_filtered_mean":  round(float(np.mean(cosines_filtered)), 4),
            "cache_cosine_filtered_std":   round(float(np.std(cosines_filtered)),  4),
            "hits_raw_at_0.95":      sum(1 for c in cosines_raw      if c >= 0.95),
            "hits_filtered_at_0.95": sum(1 for c in cosines_filtered if c >= 0.95),
            "n_total":               len(utts),
        }
        snr_curves.append(entry)

    # Pretty print
    print(f"\n{'SNR target':<14}{'raw cos':>12}{'filtered cos':>16}{'Δ':>10}{'raw hits':>11}{'filtered hits':>16}")
    print("-" * 79)
    for e in snr_curves:
        delta = e["cache_cosine_filtered_mean"] - e["cache_cosine_raw_mean"]
        print(f"  {e['snr_db_target']:>4.0f} dB        {e['cache_cosine_raw_mean']:>10.4f}  "
              f"{e['cache_cosine_filtered_mean']:>14.4f}{delta:>+10.4f}"
              f"{e['hits_raw_at_0.95']:>5}/{e['n_total']:<3}      {e['hits_filtered_at_0.95']:>5}/{e['n_total']:<3}")

    print("\nLegend:")
    print("  raw cos    = cos(fingerprint(clean), fingerprint(noisy))")
    print("  filtered cos = cos(fingerprint(clean), fingerprint(spectral-subtracted noisy))")
    print("  hits       = number of utterances where cosine ≥ 0.95 (cache would HIT)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_utterances": len(utts),
        "snr_curves":   snr_curves,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
