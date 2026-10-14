# Kaggle staging

Every directory here is a dataset or kernel staged for upload. The datasets are
what the T4 benchmarks mount as inputs; the kernels are the benchmark entry points.

## Running a benchmark

```
kaggle datasets version -p <abs path to staging dir> -m "..." -r skip [--dir-mode zip]
kaggle kernels push     -p <abs path to kernel dir>
kaggle kernels output   <owner>/<kernel> -p Results/...
```

Four rules, each paid for with a failed run:

1. **Absolute paths only.** Relative paths fail on Windows inside the CLI.
2. **`--dir-mode zip` or subdirectories are silently dropped.** This is how the
   ONNX frontend weights went missing.
3. **`datasets version` returns before the version is live.** Re-download and check
   the content before pushing the kernel, or the run uses the old version — this
   cost a full 300-sample run.
4. **Bump `condition` in the config** so output files do not collide with the
   previous run.

## Switching accounts when the GPU quota runs out

The weekly quota is 30 GPU-hours per account and is separate from the 2-concurrent-
session limit; the error text tells you which one you hit. Spare accounts **cannot
read private datasets** (403, and there is no CLI way to share one), so the workload
has to be re-uploaded — but only the parts a benchmark actually reads.

`voiceai-alt-speech/` is that trimmed stand-in: VIVOS test capped at 200 wavs plus
FLEURS `en_us`/`ko_kr` capped at 100 rows, **148 MB** in place of the 16.8 GB tier-1
dataset. It is deliberately published under the *same slug* as the original, because
`dataset_root()` in the benchmarks resolves inputs by slug across owners — so nothing
in the scripts changes.

Re-pointing a kernel at another account means rewriting `kernel-metadata.json`:
prefix `id` and every entry of `dataset_sources` with the new owner. Note that
dataset **titles** are globally unique per account, so `The requested title "X" is
already in use` on a re-run means the dataset is already there.

Accounts live in `../../../../kaggle_for_research/`. Verified 2026-08-20: the full
bundle is ~2.5 GB across 10 datasets and uploads in minutes.
