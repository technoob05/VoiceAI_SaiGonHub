"""Does forcing Simplified Chinese fix our Mandarin ASR number, or was it real?

The bidirectional T4 run measured Mandarin CER 0.2216 against English 0.0320 and
Korean 0.0732, which looked like the multilingual model simply being bad at
Mandarin. Reading the transcripts said otherwise:

    REF  这 并 不 是 告 别 ...          (FLEURS cmn_hans_cn, Simplified)
    HYP  這並不是告別 ...               (Whisper, Traditional)   CER 0.478

20 of 60 rows contained Traditional-only characters, and those scored CER 0.395
against 0.135 for the rest. So a third of the corpus was being penalised for an
orthography choice, not for hearing the wrong thing.

This is not only a scoring artifact, it is a production bug. Our MT stage is
configured for `zho_Hans`, the glossary's Mandarin column is Simplified, and the
Mandarin TTS voice expects Simplified input. A Traditional transcript degrades
every stage downstream of the ASR.

The candidate fix is Whisper's `initial_prompt`: a Simplified-Chinese prompt
biases the decoder toward Simplified output, and unlike a post-hoc character
conversion it needs no extra table, no extra dependency, and no extra latency.
This script measures whether it actually works, and what it costs.
"""

from __future__ import annotations

import io
import json
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


KAGGLE_INPUT = Path("/kaggle/input")
HERE = Path(__file__).resolve().parent
SIMPLIFIED_PROMPT = "以下是普通话的句子，请使用简体中文。"

# Characters that exist only in Traditional orthography. Not exhaustive - this is
# a detector for reporting how often the decoder drifts, not a converter.
TRADITIONAL_ONLY = set(
    "這國會來對開發時間們個學說話實現產業聽讀寫萬與專東車輪馬鳥魚龍語詞證認識關聯"
    "機構歷經濟數據網絡計劃體驗傳統應該長門問題習億圓錢銀鐵鋼張爲燈舊親愛頭風雲"
)


def dataset_root(slug: str) -> Path:
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    owners = KAGGLE_INPUT / "datasets"
    if owners.is_dir():
        for owner in sorted(owners.iterdir()):
            candidate = owner / slug
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(f"Kaggle dataset not mounted: {slug}")


def unwrap_same_name(path: Path) -> Path:
    while True:
        nested = path / path.name
        if not nested.is_dir():
            return path
        path = nested


def payload_dir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return unwrap_same_name(direct)
    matches = sorted((p for p in root.rglob(name) if p.is_dir()), key=lambda p: len(p.parts))
    if matches:
        return unwrap_same_name(matches[0])
    raise FileNotFoundError(f"Missing payload directory {name!r} under {root}")


def select_wheels(files: list[Path]) -> list[Path]:
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    chosen: dict[str, Path] = {}
    for path in sorted(files):
        parts = path.name.split("-")
        if len(parts) < 5:
            continue
        dist, tag = parts[0].lower().replace("_", "-"), parts[2]
        pure = tag.startswith("py3") or tag.startswith("py2.py3")
        if not (pure or abi in tag.split(".")):
            continue
        if dist not in chosen or not pure:
            chosen[dist] = path
    return [chosen[k] for k in sorted(chosen)]


def normalize_for_scoring(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    kept = [c if (c.isalnum() or c.isspace()) else " " for c in text]
    return " ".join("".join(kept).split())


def _levenshtein(ref: list, hyp: list) -> int:
    if not ref:
        return len(hyp)
    previous = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        current = [i]
        for j, r in enumerate(ref, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                              previous[j - 1] + (h != r)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_for_scoring(reference).replace(" ", ""))
    hyp = list(normalize_for_scoring(hypothesis).replace(" ", ""))
    return _levenshtein(ref, hyp) / max(len(ref), 1)


def main() -> int:
    wheels = sorted(dataset_root("voiceai-t4-wheels-py312").rglob("*.whl"))
    picked = select_wheels(wheels)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                    "--no-deps", *map(str, picked)], check=True)

    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel

    gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    model_dir = payload_dir(dataset_root("voiceai-t4-asr-multilingual"), "whisper-small-multilingual")
    fleurs = payload_dir(dataset_root("voiceai-t4-fleurs-zh"), "cmn_hans_cn")
    parquet = sorted(fleurs.glob("test-*.parquet"))[0]

    n = int(json.loads((dataset_root("voiceai-t4-bidir-config") / "benchmark_config.json")
                       .read_text(encoding="utf-8")).get("reverse_samples", 60))
    frame = pd.read_parquet(parquet).head(n)
    clips = []
    for _, row in frame.iterrows():
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            m = int(round(len(audio) * 16_000 / sr))
            audio = np.interp(np.linspace(0, len(audio), m, endpoint=False),
                              np.arange(len(audio)), audio).astype("float32")
        clips.append((audio, str(row["transcription"])))

    model = WhisperModel(str(model_dir), device="cuda", device_index=0,
                         compute_type="int8_float16", cpu_threads=2, num_workers=1)
    model.transcribe(clips[0][0], language="zh", beam_size=1)   # warm

    conditions = {"no_prompt": None, "simplified_prompt": SIMPLIFIED_PROMPT}
    results = {}
    for label, prompt in conditions.items():
        rows, latencies = [], []
        for index, (audio, reference) in enumerate(clips):
            t0 = time.perf_counter()
            segments, _ = model.transcribe(audio, language="zh", beam_size=1,
                                           vad_filter=True, condition_on_previous_text=False,
                                           initial_prompt=prompt)
            text = " ".join(s.text for s in segments).strip()
            latencies.append((time.perf_counter() - t0) * 1000)
            rows.append({
                "index": index, "reference": reference, "text": text,
                "cer": cer(reference, text),
                "traditional_chars": sum(ch in TRADITIONAL_ONLY for ch in text),
            })
        traditional = [r for r in rows if r["traditional_chars"] > 0]
        results[label] = {
            "n": len(rows),
            "cer_mean": statistics.fmean(r["cer"] for r in rows),
            "cer_median": statistics.median(r["cer"] for r in rows),
            "traditional_rows": len(traditional),
            "traditional_rate": len(traditional) / max(len(rows), 1),
            "cer_on_traditional_rows": (statistics.fmean(r["cer"] for r in traditional)
                                        if traditional else None),
            "cer_on_clean_rows": (statistics.fmean(r["cer"] for r in rows if r["traditional_chars"] == 0)
                                  if len(traditional) < len(rows) else None),
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_median": statistics.median(latencies),
            "rows": rows,
        }

    base, fixed = results["no_prompt"], results["simplified_prompt"]
    report = {
        "status": "complete",
        "gpu_names": gpu,
        "model": str(model_dir),
        "corpus": f"FLEURS cmn_hans_cn test, first {len(clips)} rows",
        "prompt": SIMPLIFIED_PROMPT,
        "conditions": results,
        "delta": {
            "cer": fixed["cer_mean"] - base["cer_mean"],
            "traditional_rate": fixed["traditional_rate"] - base["traditional_rate"],
            "latency_ms": fixed["latency_ms_median"] - base["latency_ms_median"],
        },
        "caveat": ("The Traditional-character detector is a non-exhaustive list, so "
                   "`traditional_rows` is a lower bound. CER is the metric that "
                   "decides this, not the detector."),
    }
    out = Path("/kaggle/working/zh_script_ab.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "conditions"},
                     ensure_ascii=False, indent=2))
    for label, block in results.items():
        print(f"{label:20s} CER={block['cer_mean']:.4f} "
              f"traditional={block['traditional_rows']}/{block['n']} "
              f"lat_median={block['latency_ms_median']:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
