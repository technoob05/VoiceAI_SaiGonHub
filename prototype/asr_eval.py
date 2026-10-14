"""asr_eval.py — batch WER/CER evaluation of our Vietnamese ASR on VIVOS test.

Answers "is the speech-to-text good enough, and do we need a spelling/diacritic
post-corrector?" with numbers, not vibes. Also breaks errors into:
  * diacritic/tone errors — hyp == ref after stripping Vietnamese diacritics but
    differs with them. This is THE critical Vietnamese error class (tone = meaning)
    and the one a post-corrector or the tone-aware-quantization spine targets.
  * word substitutions/insertions/deletions — genuine content errors.

Usage:
    python asr_eval.py --n 40                 # 40 random test clips
    python asr_eval.py --n 100 --seed 1
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VIVOS = HERE.parent / "data" / "processed" / "vivos" / "vivos" / "test"


def norm(s: str) -> str:
    """Lowercase, drop punctuation, collapse spaces. Keep Vietnamese diacritics."""
    s = s.lower().strip()
    s = re.sub(r"[.,;:!?…\"'`()\[\]{}\-–—]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_diacritics(s: str) -> str:
    """Remove Vietnamese tone/vowel diacritics (đ→d) for the tone-error diff."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token lists (for WER/CER)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h), len(r)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return edit_distance(r, h), len(r)


def load_prompts() -> dict:
    out = {}
    with open(VIVOS / "prompts.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def wav_path(clip_id: str) -> Path:
    speaker = clip_id.split("_")[0]
    return VIVOS / "waves" / speaker / f"{clip_id}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--worst", type=int, default=8, help="show N worst examples")
    ap.add_argument("--beam", type=int, default=None, help="override ASR beam size")
    ap.add_argument("--model-dir", default=None, help="override ASR CT2 model dir")
    ap.add_argument("--prompt", default=None, help="initial_prompt for domain priming")
    args = ap.parse_args()

    import soundfile as sf
    from pipeline_fast import FastCascadeS2ST, FastPipelineConfig

    prompts = load_prompts()
    ids = [i for i in prompts if wav_path(i).exists()]
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]

    print(f"[asr_eval] {len(ids)} VIVOS test clips  (seed={args.seed})")
    cfg = FastPipelineConfig()
    if args.beam is not None:
        cfg.asr_beam = args.beam
    if args.model_dir is not None:
        cfg.asr_ct2_dir = args.model_dir
    if args.prompt is not None:
        cfg.asr_prompt = args.prompt
    pipe = FastCascadeS2ST(cfg)
    pipe._ensure_asr()
    print(f"[asr_eval] model={Path(pipe.cfg.asr_ct2_dir).name}  beam={pipe.cfg.asr_beam}")

    W_err = W_tot = C_err = C_tot = 0
    tone_only = exact = 0
    rows = []
    t0 = time.time()
    for k, cid in enumerate(ids, 1):
        audio, sr = sf.read(wav_path(cid), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        hyp_raw = pipe.transcribe(audio, sr)
        ref, hyp = norm(prompts[cid]), norm(hyp_raw)
        we, wt = wer(ref, hyp)
        ce, ct = cer(ref, hyp)
        W_err += we; W_tot += wt; C_err += ce; C_tot += ct
        clip_wer = we / max(wt, 1)
        if ref == hyp:
            exact += 1
        elif strip_diacritics(ref) == strip_diacritics(hyp):
            tone_only += 1   # only the diacritics/tones differ
        rows.append((clip_wer, cid, ref, hyp))
        if k % 10 == 0:
            print(f"  {k}/{len(ids)} … running WER={W_err/max(W_tot,1):.3f}", flush=True)

    print("\n" + "=" * 70)
    print(f"  Clips: {len(ids)}   ({time.time()-t0:.0f}s)")
    print(f"  WER = {W_err/max(W_tot,1)*100:.2f}%   CER = {C_err/max(C_tot,1)*100:.2f}%")
    print(f"  Exact match: {exact}/{len(ids)} ({exact/len(ids)*100:.0f}%)")
    print(f"  Diacritic/TONE-only errors: {tone_only}/{len(ids)} "
          f"({tone_only/len(ids)*100:.0f}%)  ← post-corrector / tone-aware target")
    print("=" * 70)
    import json
    pairs = [{"id": cid, "ref": ref, "hyp": hyp} for (_, cid, ref, hyp) in rows]
    json.dump(pairs, open(f"asr_eval_pairs_beam{pipe.cfg.asr_beam}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    rows.sort(reverse=True)
    print(f"\nWorst {args.worst} clips:")
    for clip_wer, cid, ref, hyp in rows[: args.worst]:
        tag = "TONE" if strip_diacritics(ref) == strip_diacritics(hyp) else "WORD"
        print(f"  [{clip_wer*100:5.1f}% {tag}] {cid}")
        print(f"     ref: {ref}")
        print(f"     hyp: {hyp}")


if __name__ == "__main__":
    main()
