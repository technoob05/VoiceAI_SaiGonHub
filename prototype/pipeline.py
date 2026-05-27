"""
Vietnamese-centric cascaded Speech-to-Speech Translation skeleton.

Stage 1 — ASR:  PhoWhisper-small        (Vi audio -> Vi text)
Stage 2 — MT:   NLLB-200-distilled-600M (Vi text  -> En/Zh/Ko text)
Stage 3 — TTS:  MMS-TTS (VITS)          (target text -> target audio)

All checkpoints public, downloaded once into <repo>/data/raw/hf_cache/ on first run.
Designed for local CPU dev (slow but functional ~10 s per 5 s utterance).
Replace any single stage with our tone-aware quantized variant later — that's the spine.

Quick start (after `pip install transformers soundfile torch`):

    python prototype/pipeline.py \\
        --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \\
        --target en \\
        --out prototype/out_en.wav

    # try all three target languages on the same input:
    python prototype/pipeline.py --audio <vi.wav> --target en --out out_en.wav
    python prototype/pipeline.py --audio <vi.wav> --target zh --out out_zh.wav
    python prototype/pipeline.py --audio <vi.wav> --target ko --out out_ko.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Redirect ALL HF caches (incl. xet staging) to <repo>/data/raw/hf_cache BEFORE importing
# transformers. Critical on Windows when C: drive is full — xet defaults to %TEMP% on C:
# and silently fails with "There is not enough space on the disk."
PROJECT = Path(__file__).resolve().parents[1]
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
(HF_CACHE / "xet").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME",            str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",       str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE",       str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

# Ensure stdout can print Vietnamese on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try: _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                          # noqa: E402
import soundfile as sf                                      # noqa: E402
import torch                                                # noqa: E402
from transformers import (                                  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    NllbTokenizerFast,
    VitsModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# NLLB-200 language codes (BCP-47 + script).
NLLB_CODE = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}
# Per-target MMS-TTS checkpoints.
TTS_REPO = {
    "vi": "facebook/mms-tts-vie",
    "en": "facebook/mms-tts-eng",
    "zh": "facebook/mms-tts-cmn",
    "ko": "facebook/mms-tts-kor",
}


@dataclass
class PipelineConfig:
    asr_id:  str = "vinai/PhoWhisper-small"
    mt_id:   str = "facebook/nllb-200-distilled-600M"
    device:  str = "cpu"               # "cpu" / "cuda" / "mps"
    asr_lang: str = "vi"               # forced decoder language for Whisper
    max_new_tokens: int = 256


class CascadeS2ST:
    """Three-stage cascade. Each stage is lazily loaded; TTS is per-language lazy."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self._asr_proc:  WhisperProcessor | None = None
        self._asr_model: WhisperForConditionalGeneration | None = None
        self._mt_tok = None
        self._mt_model = None
        self._tts_cache: dict[str, tuple] = {}

    # ---------------- Stage 1: ASR (PhoWhisper) ----------------
    def _ensure_asr(self) -> None:
        if self._asr_model is not None:
            return
        print(f"[load] ASR: {self.cfg.asr_id}")
        self._asr_proc = WhisperProcessor.from_pretrained(self.cfg.asr_id)
        m = WhisperForConditionalGeneration.from_pretrained(self.cfg.asr_id)
        m.generation_config.language = self.cfg.asr_lang
        m.generation_config.task = "transcribe"
        self._asr_model = m.to(self.cfg.device).eval()

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        self._ensure_asr()
        if sr != 16000:
            # PhoWhisper expects 16 kHz; resample crudely via numpy interp if needed.
            target = int(round(len(audio) * 16000 / sr))
            audio = np.interp(np.linspace(0, len(audio), target, endpoint=False),
                              np.arange(len(audio)), audio).astype(np.float32)
            sr = 16000
        inputs = self._asr_proc(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = self._asr_model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
        return self._asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ---------------- Stage 2: MT (NLLB-200-distilled-600M) ----------------
    def _ensure_mt(self) -> None:
        if self._mt_model is not None:
            return
        print(f"[load] MT:  {self.cfg.mt_id}")
        # NLLB requires explicit NllbTokenizerFast — AutoTokenizer factory has a known
        # broken lookup in transformers 5.0.0 for the m2m_100 -> NllbTokenizer mapping.
        if "nllb" in self.cfg.mt_id.lower():
            self._mt_tok = NllbTokenizerFast.from_pretrained(self.cfg.mt_id)
        else:
            self._mt_tok = AutoTokenizer.from_pretrained(self.cfg.mt_id)
        self._mt_model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.mt_id).to(self.cfg.device).eval()

    def translate(self, text: str, target: str, source: str = "vi") -> str:
        if not text.strip():
            return ""
        self._ensure_mt()
        self._mt_tok.src_lang = NLLB_CODE[source]
        inputs = self._mt_tok(text, return_tensors="pt").to(self.cfg.device)
        tgt_bos_id = self._mt_tok.convert_tokens_to_ids(NLLB_CODE[target])
        with torch.inference_mode():
            ids = self._mt_model.generate(
                **inputs,
                forced_bos_token_id=tgt_bos_id,
                max_new_tokens=self.cfg.max_new_tokens,
            )
        return self._mt_tok.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ---------------- Stage 3: TTS (MMS-TTS / VITS) ----------------
    def _ensure_tts(self, target: str):
        if target in self._tts_cache:
            return self._tts_cache[target]
        repo = TTS_REPO[target]
        print(f"[load] TTS: {repo}")
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo).to(self.cfg.device).eval()
        self._tts_cache[target] = (tok, model)
        return self._tts_cache[target]

    def synthesize(self, text: str, target: str) -> tuple[np.ndarray, int]:
        tok, model = self._ensure_tts(target)
        if not text.strip():
            return np.zeros(0, dtype=np.float32), model.config.sampling_rate
        inputs = tok(text, return_tensors="pt").to(self.cfg.device)
        with torch.inference_mode():
            wav = model(**inputs).waveform
        return wav.squeeze().cpu().numpy().astype(np.float32), model.config.sampling_rate

    # ---------------- End-to-end ----------------
    def run(self, audio: np.ndarray, sr: int, target: str) -> dict:
        t0 = time.perf_counter()
        vi_text = self.transcribe(audio, sr)
        t1 = time.perf_counter()
        tgt_text = self.translate(vi_text, target, source="vi")
        t2 = time.perf_counter()
        tgt_audio, out_sr = self.synthesize(tgt_text, target)
        t3 = time.perf_counter()
        return {
            "vi_text":   vi_text,
            "tgt_text":  tgt_text,
            "tgt_audio": tgt_audio,
            "sr":        out_sr,
            "latency":   {"asr_ms": (t1 - t0) * 1000,
                          "mt_ms":  (t2 - t1) * 1000,
                          "tts_ms": (t3 - t2) * 1000,
                          "total_ms": (t3 - t0) * 1000},
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio",  required=True,                    help="Input WAV (Vietnamese speech)")
    p.add_argument("--target", choices=["en", "zh", "ko", "vi"], default="en", help="Target language")
    p.add_argument("--out",    default="prototype/out.wav",       help="Output WAV path")
    p.add_argument("--device", default="cpu",                     help='"cpu" | "cuda" | "mps"')
    p.add_argument("--asr-id", default="vinai/PhoWhisper-small",
                   help="HF id for ASR (any Whisper-family checkpoint)")
    p.add_argument("--mt-id",  default="facebook/nllb-200-distilled-600M",
                   help="HF id for MT (any NLLB-family checkpoint)")
    args = p.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"[input] {args.audio}  duration={duration:.2f}s  sr={sr}")

    cfg = PipelineConfig(asr_id=args.asr_id, mt_id=args.mt_id, device=args.device)
    pipe = CascadeS2ST(cfg)
    res = pipe.run(audio, sr, args.target)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), res["tgt_audio"], res["sr"])
    out_dur = len(res["tgt_audio"]) / max(res["sr"], 1)

    print()
    print(f"[asr]   vi : {res['vi_text']}")
    print(f"[mt]    {args.target:>2} : {res['tgt_text']}")
    print(f"[tts]   wrote {out}  ({out_dur:.2f}s @ {res['sr']} Hz)")
    print()
    lat = res["latency"]
    print(f"[latency] asr={lat['asr_ms']:>7.0f} ms   mt={lat['mt_ms']:>7.0f} ms   tts={lat['tts_ms']:>7.0f} ms   total={lat['total_ms']:>7.0f} ms")
    rtf = lat["total_ms"] / 1000 / duration
    print(f"[rtf]   real-time factor = {rtf:.2f}x  (>1 means slower than real-time on this device)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
