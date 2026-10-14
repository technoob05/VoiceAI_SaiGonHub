"""Download and verify the Vietnamese corrector for later offline inference.

This is the only step that needs network access. The live application always
loads the resulting checkpoint with ``local_files_only=True``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))
sys.path.insert(0, str(HERE))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bmd1905/vietnamese-correction-v2")
    parser.add_argument("--no-smoke", action="store_true",
                        help="download files without loading the model")
    parser.add_argument("--int8", action="store_true",
                        help="also test dynamic INT8 loading (slower preparation)")
    parser.add_argument("--ct2", action="store_true",
                        help="export a fast CTranslate2 INT8 corrector")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] downloading {args.model} into {HF_CACHE}")
    # The upstream repository also contains deeply nested TensorBoard logs whose
    # path exceeds the legacy Windows path limit. They are irrelevant to inference.
    snapshot = snapshot_download(
        repo_id=args.model,
        cache_dir=HF_CACHE / "hub",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "sentencepiece.model",
            "dict.txt",
            "README.md",
        ],
    )
    print(f"[prepare] snapshot ready: {snapshot}")

    if args.ct2:
        from transformers import MBartConfig

        for item in dataclasses.fields(MBartConfig):
            if item.name == "scale_embedding" and isinstance(item.default, bool):
                item.default = int(item.default)

        source = Path(snapshot)
        compat = PROJECT / "data" / "raw" / "corrector_hf_compat"
        output = PROJECT / "data" / "raw" / "models_ct2" / "vietnamese-correction-v2-int8"
        compat.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "dict.txt", "sentencepiece.bpe.model",
                     "special_tokens_map.json", "tokenizer_config.json"):
            src = source / name
            dst = compat / name
            if dst.exists():
                continue
            try:
                os.link(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config["scale_embedding"] = int(config.get("scale_embedding", 0))
        # CT2's MBart loader still reads this legacy transformers attribute.
        config.setdefault("normalize_before", True)
        (compat / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        from ctranslate2.converters import TransformersConverter
        print(f"[prepare] exporting CT2 INT8 to {output}")
        TransformersConverter(
            str(compat),
            copy_files=["dict.txt", "sentencepiece.bpe.model",
                        "special_tokens_map.json", "tokenizer_config.json"],
        ).convert(str(output), quantization="int8", force=True)
        print(f"[prepare] CT2 model ready: {output}")

    if args.no_smoke:
        return 0

    from transcript_corrector import CorrectorConfig, TranscriptCorrector

    corrector = TranscriptCorrector(CorrectorConfig(
        model=args.model,
        local_files_only=True,
        dynamic_int8=args.int8,
    ))
    if not corrector.prewarm():
        print(f"[error] checkpoint downloaded but local model load failed: "
              f"{corrector._load_error}", file=sys.stderr)
        return 2
    sample = "Tôi đang học AI ở trun tâm AI Việt Nam"
    result = corrector.correct(sample)
    print(f"[smoke] raw:       {result.raw_text}")
    print(f"[smoke] proposed:  {result.proposed_text}")
    print(f"[smoke] selected:  {result.text}")
    print(f"[smoke] accepted={result.accepted} reason={result.reason} "
          f"latency_ms={result.latency_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
