#!/usr/bin/env python3
"""parakeet_vi_eval.py — run NVIDIA Parakeet-CTC-0.6B-Vietnamese and measure WER.

⚠️ RUN THIS ON A LINUX BOX (GreenNode pod / Colab), Python 3.10–3.12. It needs
NeMo, which does NOT build on Windows / Python 3.14 (ctc_segmentation C++ wheel
fails — verified 2026-07-14). That is why it is a standalone script, not wired
into the Windows pipeline.

Why Parakeet is the bake-off candidate (see DEPLOYMENT_ROADMAP.md / the SOTA survey):
  * CTC (FastConformer-CTC), not autoregressive → one-pass decode, low/predictable
    latency, NO hallucination failure mode, quantizes to INT8 gracefully.
  * 2000h Vietnamese training (vs PhoWhisper's 844h) → better acoustic coverage,
    which targets our dominant error class (n/l, ph/v, x/s, r/g, numbers).
  * Reported VIVOS 5.96 / FLEURS 5.15. Commercial use permitted.
  * sherpa-onnx (our runtime) runs NeMo CTC via OfflineNemoEncDecCtcModelConfig,
    so after a one-time .nemo→ONNX export it drops into our on-device stack.
  * BONUS: the HF repo ships a KenLM binary + lexicon → free shallow-fusion LM
    (the "LM rescoring" lever) without training one.

Setup (on the Linux pod):
    pip install "nemo_toolkit[asr]" soundfile
    # optional, for the WER comparison to be apples-to-apples with our INT8 numbers,
    # also run our PhoWhisper eval (prototype/asr_eval.py) on the SAME clips.

Usage:
    python parakeet_vi_eval.py --vivos /path/to/vivos/test --n 40 --seed 42
    python parakeet_vi_eval.py --vivos ... --export-onnx parakeet_vi_onnx/   # for sherpa-onnx
"""
from __future__ import annotations

import argparse
import random
import re
import time
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---- self-contained WER/CER (same normalization as prototype/asr_eval.py) ----
def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[.,;:!?…\"'`()\[\]{}\-–—]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_diacritics(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _edit(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return _edit(r, h), len(r)


def cer(ref, hyp):
    return _edit(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))), len(ref.replace(" ", ""))


def load_prompts(vivos: Path) -> dict:
    out = {}
    with open(vivos / "prompts.txt", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(" ", 1)
            if len(p) == 2:
                out[p[0]] = p[1]
    return out


def wav_path(vivos: Path, cid: str) -> Path:
    return vivos / "waves" / cid.split("_")[0] / f"{cid}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vivos", required=True, help="path to vivos/test (has prompts.txt + waves/)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="nvidia/parakeet-ctc-0.6b-Vietnamese")
    ap.add_argument("--export-onnx", default=None, help="also export .nemo→ONNX into this dir")
    ap.add_argument("--worst", type=int, default=8)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    print(f"[parakeet] loading {args.model} (NeMo) …")
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()

    if args.export_onnx:
        out = Path(args.export_onnx); out.mkdir(parents=True, exist_ok=True)
        onnx_path = str(out / "model.onnx")
        print(f"[parakeet] exporting ONNX → {onnx_path} (for sherpa-onnx OfflineNemoEncDecCtcModelConfig)")
        model.export(onnx_path)
        # sherpa-onnx also needs tokens.txt: dump the CTC vocabulary.
        with open(out / "tokens.txt", "w", encoding="utf-8") as f:
            for i, tok in enumerate(model.decoder.vocabulary):
                f.write(f"{tok} {i}\n")
            f.write(f"<blk> {len(model.decoder.vocabulary)}\n")
        print("[parakeet] ONNX + tokens.txt written. Run via sherpa_onnx.OfflineRecognizer.from_nemo_ctc(...).")

    vivos = Path(args.vivos)
    prompts = load_prompts(vivos)
    ids = [i for i in prompts if wav_path(vivos, i).exists()]
    random.Random(args.seed).shuffle(ids)
    ids = ids[: args.n]
    paths = [str(wav_path(vivos, i)) for i in ids]

    print(f"[parakeet] transcribing {len(ids)} VIVOS clips …")
    t0 = time.time()
    hyps = model.transcribe(paths, batch_size=8)
    # NeMo may return list[str] or list[Hypothesis]
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    dt = time.time() - t0

    W_err = W_tot = C_err = C_tot = exact = tone = 0
    rows = []
    for cid, hyp_raw in zip(ids, hyps):
        ref, hyp = norm(prompts[cid]), norm(hyp_raw)
        we, wt = wer(ref, hyp); ce, ct = cer(ref, hyp)
        W_err += we; W_tot += wt; C_err += ce; C_tot += ct
        if ref == hyp:
            exact += 1
        elif strip_diacritics(ref) == strip_diacritics(hyp):
            tone += 1
        rows.append((we / max(wt, 1), cid, ref, hyp))

    print("\n" + "=" * 70)
    print(f"  Parakeet-CTC-0.6B-vi  |  {len(ids)} clips  ({dt:.0f}s, {dt/len(ids):.2f}s/clip)")
    print(f"  WER = {W_err/max(W_tot,1)*100:.2f}%   CER = {C_err/max(C_tot,1)*100:.2f}%")
    print(f"  Exact = {exact}/{len(ids)} ({exact/len(ids)*100:.0f}%)   tone-only = {tone}/{len(ids)}")
    print("  Compare to (our INT8 on same clips): PhoWhisper-small 10.63% / medium 7.97%")
    print("=" * 70)
    rows.sort(reverse=True)
    print(f"\nWorst {args.worst}:")
    for w, cid, ref, hyp in rows[: args.worst]:
        print(f"  [{w*100:5.1f}%] {cid}\n     ref: {ref}\n     hyp: {hyp}")


if __name__ == "__main__":
    main()
