# exp/ — Experiment scripts

Each `.py` file in this folder is a **self-contained Kaggle-ready script**: copy it, paste into a Kaggle notebook (or upload as a script), run end-to-end. No external imports from a shared `src/` package — duplicate any helper you need into the file. This is intentional: experiments rot, reproducibility wins.

## Read first
- [tracker.md](tracker.md) — current SOTA, active scripts, methodology rules. **Read before writing a new exp.**
- [../CLAUDE.md](../CLAUDE.md) — project plan, research spine, novelty defense.

## File-naming
```
expNN_<verb>_<object>.py
```
- `NN` = zero-padded two-digit (use 3 digits once we cross 99)
- `<verb>` = action: `probe`, `quantize`, `distill`, `train`, `eval`, `bench`
- `<object>` = subject: `phowhisper_vi`, `int4_uniform`, `tone_layers`

Examples: `exp04_probe_tone_layers.py`, `exp05_quantize_selective_v1.py`.

## Self-contained template (each new exp should look like this)
```python
"""
expNN — <one-line description>
Spine: tone-aware mixed-precision quantization for Vietnamese S2ST.
Hardware: Kaggle T4 / P100 / local.
Result file: Results/expNN_<name>.json
"""
from dataclasses import dataclass, asdict
import json, time, hashlib, os
from pathlib import Path

@dataclass
class Config:
    EXP_ID: str = "expNN"
    MODEL_ID: str = "vinai/PhoWhisper-small"
    DATASET_ID: str = "vivos"            # or a path to a Kaggle dataset
    SPLIT: str = "test"
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2                  # Kaggle CPU is small
    QUANT_CONFIG: str = "fp32"           # 'fp32' | 'int8_uniform' | 'int4_uniform' | 'selective_v1' | ...
    SEED: int = 42
    OUTPUT_DIR: str = "Results"

cfg = Config()

# ---- Load model + dataset (Kaggle-friendly: handle offline cache) ----
# ...

# ---- Quantize ----
# ...

# ---- Eval loop: collect predictions, references, per-utterance latency ----
# ...

# ---- Metrics: WER, TER, BLEU/chrF/COMET (if applicable), latency p50/p95/p99,
#       model_size_mb, peak_memory_mb, bootstrap CI ----
# ...

# ---- Save ----
result = {
    "config": asdict(cfg),
    "metrics": {
        "wer": ..., "ter": ..., "wer_ci95": [..., ...],
        "latency_p50_ms": ..., "latency_p95_ms": ..., "latency_p99_ms": ...,
        "model_size_mb": ..., "peak_memory_mb": ...,
        "n_test_utterances": ...,
    },
    "git_sha": os.environ.get("GIT_SHA", "n/a"),
    "wall_clock_seconds": ...,
}
Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
out = Path(cfg.OUTPUT_DIR) / f"{cfg.EXP_ID}_{cfg.QUANT_CONFIG}.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"[done] wrote {out}")
```

## Kaggle workflow
1. Make sure the Kaggle Dataset for our raw data is uploaded (see `../data/README.md` and `../kaggle_uploads/`).
2. Create a new Kaggle Notebook → "Add Data" → attach the dataset.
3. Paste the exp script. Make sure paths point to `/kaggle/input/<dataset-slug>/...`.
4. Set "Internet on" only if the script needs to download a model from HuggingFace at runtime (otherwise pre-cache the model into the dataset).
5. Run, wait, download `Results/expNN_*.json`.
6. Commit the JSON to this repo. Update [tracker.md](tracker.md).

## Legacy
`exp/legacy/` is the archive for experiments that have been superseded. Keep around for reproducibility, don't run.

## What does NOT belong in exp/
- Library code (helpers used across scripts) — duplicate into each script instead. Painful but reproducible.
- Documentation about the method — that lives in [../CLAUDE.md](../CLAUDE.md) and [../papers/](../papers/).
- Notebooks (`.ipynb`) — convert to `.py` before committing. Notebooks diff poorly.
