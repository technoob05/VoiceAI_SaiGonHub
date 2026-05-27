# data/download/ — Dataset download + Kaggle bundle scripts

Two scripts, run in order:

## 1. `download_tier1.py` — HuggingFace → local cache
```bash
python data/download/download_tier1.py --cache-dir ~/.cache/huggingface
```
Pulls ~45 GB of redistributable Tier 1 datasets (see [../README.md](../README.md) §Priority tiers) into the standard HuggingFace cache. Takes hours on a normal connection.

Flags:
- `--cache-dir PATH` — override HF cache location.
- `--skip ID1 ID2` — skip specific dataset IDs (e.g. `--skip AISHELL/AISHELL-3` if you already have it).
- `--dry-run` — print plan without downloading.

What it does NOT download (and why):
- Common Voice 17 — gated by Mozilla acceptance flow; do `huggingface-cli login` + accept terms in the browser, then add it manually.
- VLSP, PhoMT, PhoST, MuST-C, KsponSpeech — research-only licenses; not for redistribution. Download separately if you need them for ablations.

## 2. `build_kaggle_bundles.py` — zip → 3 Kaggle datasets
```bash
python data/download/build_kaggle_bundles.py --cache-dir ~/.cache/huggingface --out kaggle_uploads/
```
Walks the HF cache, zips Tier 1 contents into three `.zip` files (each ≤ 20 GB so no Kaggle phone-verification needed):

| Bundle | Contents | Size |
|---|---|---|
| `voiceai-tier1-vi.zip` | VIVOS + FLEURS-vi + InfoRe + FLORES-plus | ~8 GB |
| `voiceai-tier1-zh.zip` | AISHELL-3 + FLEURS-cmn | ~20 GB |
| `voiceai-tier1-ko-en.zip` | Zeroth-Korean + FLEURS-ko + FLEURS-en | ~8 GB |

Each zip carries a `LICENSES.md` (attribution) and a `MANIFEST.json` (file list, sizes, optional SHA256).

Flags:
- `--only voiceai-tier1-vi` — build only one bundle.
- `--hash` — compute SHA256 per file in MANIFEST (slow but reproducibility-grade).

## 3. Upload to Kaggle (manual or CLI)

**Web UI** (simpler): Kaggle → New Dataset → Upload Zip → drag `kaggle_uploads/voiceai-tier1-vi.zip` → set slug `voiceai-tier1-vi` → Public.

**CLI** (faster, for the other two):
```bash
pip install kaggle    # one-time
# ~/.kaggle/kaggle.json must be in place
cd kaggle_uploads
kaggle datasets init -p voiceai-tier1-zh
# edit voiceai-tier1-zh/dataset-metadata.json (title, id, licenses)
kaggle datasets create -p voiceai-tier1-zh
```

After all three are public, attach them to any Kaggle notebook via "+ Add Data".

## 4. What's NOT in here

- Industrial-noise Vietnamese speech corpus does not exist publicly. Synthesis pipeline (DEMAND `PSTATION`+`DKITCHEN` + MUSAN at SNR ∈ {0, 5, 10, 15} dB over VIVOS / FLEURS-vi) lives in a separate experiment script — see [../../exp/](../../exp/).
- Forced alignments for tonal phoneme labels (needed for the Tone-Error-Rate metric) are produced by `exp00_ter_metric.py`, not pre-computed here.
