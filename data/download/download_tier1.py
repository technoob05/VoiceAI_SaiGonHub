"""
Download Tier 1 datasets via huggingface_hub.snapshot_download (resilient to loading-script removal).

WHY snapshot_download instead of load_dataset:
  datasets 4.4.x removed support for dataset loading scripts (e.g. vivos.py, fleurs.py).
  load_dataset() throws "Dataset scripts are no longer supported" on legacy datasets.
  snapshot_download pulls raw files (audio + transcripts) directly from the HF repo,
  bypassing the parser. Each experiment script (exp/exp00 etc.) then walks the files
  using its own loader. This is more portable to Kaggle too.

Run from project root:
    python data/download/download_tier1.py --cache-dir data/raw/hf_cache

Phase A (~6 GB, default — skip AISHELL-3 + Zeroth-Korean):
    --skip "AISHELL/AISHELL-3" "kresnik/zeroth_korean"

Gated datasets (must `huggingface-cli login` first AND accept terms in browser):
    openlanguagedata/flores_plus

After download, raw files live at data/raw/hf_cache/datasets--<owner>--<name>/snapshots/<sha>/...
(plus convenience symlinks under data/raw/hf_cache/<owner>__<name>/).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

# (repo_id, allow_patterns or None, note, requires_auth)
TIER1: list[tuple[str, list[str] | None, str, bool]] = [
    ("AILAB-VNUHCM/vivos",           None,                    "Vietnamese ASR (15h, 6 tones)",          False),
    ("google/fleurs",                ["vi_vn/**", "*.md"],    "FLEURS Vietnamese",                       False),
    ("google/fleurs",                ["cmn_hans_cn/**"],      "FLEURS Mandarin",                         False),
    ("google/fleurs",                ["ko_kr/**"],            "FLEURS Korean (non-tonal CONTROL)",       False),
    ("google/fleurs",                ["en_us/**"],            "FLEURS English",                          False),
    ("openlanguagedata/flores_plus", None,                    "FLORES-plus MT eval (text)  [GATED]",     True),
    ("doof-ferb/infore1_25hours",    None,                    "Vietnamese TTS single-speaker 25h",       False),
]

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=None, help="HF cache dir (default: ~/.cache/huggingface)")
    p.add_argument("--skip",      nargs="*", default=[], help="Repo IDs to skip")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--max-workers", type=int, default=8, help="Concurrent file downloads per dataset")
    args = p.parse_args()

    if args.cache_dir:
        cache = Path(args.cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
        print(f"[info] HF cache -> {cache}")

    print(f"[plan] {len(TIER1)} datasets, skip={args.skip or 'none'}")
    for repo, patterns, note, gated in TIER1:
        marker = " [SKIP]" if repo in args.skip else (" [GATED-AUTH-REQUIRED]" if gated else "")
        sub = f" patterns={patterns}" if patterns else ""
        print(f"  - {repo}{sub} - {note}{marker}")
    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    failures: list[tuple[str, str]] = []
    for repo, patterns, note, gated in TIER1:
        if repo in args.skip:
            continue
        tag = f"{repo}" + (f" {patterns}" if patterns else "")
        t0 = time.time()
        print(f"[download] {tag} - {note}")
        try:
            local = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=patterns,
                max_workers=args.max_workers,
            )
            print(f"[ok]       {tag} in {time.time()-t0:.0f}s -> {local}")
        except GatedRepoError as e:
            print(f"[GATED]    {tag}: must `huggingface-cli login` and accept terms at https://huggingface.co/datasets/{repo}")
            failures.append((tag, "gated; auth required"))
        except RepositoryNotFoundError as e:
            print(f"[404]      {tag}: repo not found or private; {e}")
            failures.append((tag, "404"))
        except Exception as e:
            print(f"[FAIL]     {tag}: {type(e).__name__}: {e}")
            failures.append((tag, str(e)))

    if failures:
        print(f"\n[summary] {len(failures)} failures:")
        for tag, err in failures:
            print(f"  - {tag}: {err}")
        return 1
    print("\n[summary] all Tier 1 downloads succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
