"""
Bundle Tier 1 datasets into Kaggle-compatible .zip files (each <= 20 GB so no phone verification needed).

Run from project root AFTER:
  1) data/download/download_tier1.py has populated data/raw/hf_cache/
  2) VIVOS has been extracted to data/processed/vivos/

Produces (under --out):
    voiceai-tier1-vi.zip       ~12 GB  VIVOS (extracted) + FLEURS-vi + InfoRe-25h Vi TTS
    voiceai-tier1-zh.zip       ~2.5 GB FLEURS-cmn  (extend with AISHELL-3 in Phase B)
    voiceai-tier1-ko-en.zip    ~5 GB   FLEURS-ko + FLEURS-en  (extend with Zeroth-Ko in Phase B)

Each zip carries:
    LICENSES.md   - per-source attribution + license text
    MANIFEST.json - list of files, sizes, optional sha256
    README.md     - Kaggle-side layout + how to use in a notebook

Inside each zip files use the layout that exp/exp00_ter_metric.py and friends expect:
    voiceai-tier1-vi/
      vivos/{train,test}/{waves/<spk>/<utt>.wav, prompts.txt, genders.txt}
      fleurs/vi_vn/{train,test,validation}-00000-of-00001.parquet
      infore_25h/train-*-of-15.parquet  (15 shards)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]   # ../../

# (zip_name, [(source_path_relative_to_PROJECT, arcname_inside_zip)])
BUNDLES: dict[str, list[tuple[str, str]]] = {
    "voiceai-tier1-vi": [
        ("data/processed/vivos/vivos",                                              "vivos"),
        ("data/raw/hf_cache/hub/datasets--google--fleurs/.GLOB/parquet-data/vi_vn", "fleurs/vi_vn"),
        ("data/raw/hf_cache/hub/datasets--doof-ferb--infore1_25hours/.GLOB/data",   "infore_25h"),
    ],
    "voiceai-tier1-zh": [
        ("data/raw/hf_cache/hub/datasets--google--fleurs/.GLOB/parquet-data/cmn_hans_cn", "fleurs/cmn_hans_cn"),
    ],
    "voiceai-tier1-ko-en": [
        ("data/raw/hf_cache/hub/datasets--google--fleurs/.GLOB/parquet-data/ko_kr", "fleurs/ko_kr"),
        ("data/raw/hf_cache/hub/datasets--google--fleurs/.GLOB/parquet-data/en_us", "fleurs/en_us"),
    ],
}

LICENSES_TXT = """# License & attribution

This Kaggle dataset bundles publicly redistributable speech and text corpora. Each upstream source
retains its own license; please cite them in any derivative work. We re-host only datasets whose
licenses permit redistribution.

## Sources

- **VIVOS** - Luong, Hieu Thi and Hai Quan Vu. "A non-expert Kaldi recipe for Vietnamese Speech
  Recognition System." WLSI/OIAF4HLT 2016. CC-BY-SA-4.0.
- **FLEURS** - Conneau et al. "FLEURS: Few-shot Learning Evaluation of Universal Representations
  of Speech." IEEE SLT 2022. CC-BY-4.0.
- **InfoRe 25h Vietnamese TTS** - InfoRe Technology, redistributed by `doof-ferb`. CC-BY-4.0.

For research and reproducibility only. Attribute every source above when publishing derivatives.
"""

README_TPL = """# {bundle}

Kaggle bundle from VoiceAI_SaiGonHub. Attach this dataset to a Kaggle notebook and load via:

```python
import pyarrow.parquet as pq
import soundfile as sf
from pathlib import Path

ROOT = Path('/kaggle/input/{bundle}')

# FLEURS (parquet, audio bytes embedded)
table = pq.read_table(ROOT / 'fleurs' / 'vi_vn' / 'test-00000-of-00001.parquet')
for row in table.to_pylist():
    import io
    audio, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
    text = row.get('transcription') or row.get('raw_transcription')
    ...

# VIVOS (extracted WAV + prompts.txt)
for line in (ROOT / 'vivos' / 'test' / 'prompts.txt').read_text(encoding='utf-8').splitlines():
    utt_id, _, text = line.partition(' ')
    spk = utt_id.split('_')[0]
    wav = ROOT / 'vivos' / 'test' / 'waves' / spk / (utt_id + '.wav')
    audio, sr = sf.read(str(wav), dtype='float32')
    ...
```

See `LICENSES.md` for attribution and `MANIFEST.json` for the full file list.
"""


def _resolve_glob_path(p: Path) -> Path | None:
    """Resolve ".GLOB" placeholders by picking the only matching directory.
    Used to find the unique HF snapshot SHA dir without hard-coding it.
    """
    parts = p.parts
    if ".GLOB" not in parts:
        return p if p.exists() else None
    idx = parts.index(".GLOB")
    prefix = Path(*parts[:idx])
    suffix = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path()
    if not (prefix / "snapshots").is_dir():
        return None
    snaps = sorted((prefix / "snapshots").iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    for snap in snaps:
        cand = snap / suffix if suffix.parts else snap
        if cand.exists():
            return cand
    return None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _zip_one(bundle: str, srcs: list[tuple[str, str]], out_dir: Path, do_hash: bool) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{bundle}.zip"
    if out.exists():
        print(f"[skip] {out} already exists; delete to rebuild")
        return out

    resolved: list[tuple[Path, str]] = []
    for raw, arc in srcs:
        p = _resolve_glob_path((PROJECT / raw).resolve())
        if p is None or not p.exists():
            print(f"  [warn] missing source: {raw}  ->  {p}")
            continue
        resolved.append((p, arc))
    if not resolved:
        print(f"  [skip] {bundle}: no sources resolved")
        return None

    print(f"[zip ] {out}")
    total_uncompressed = 0
    for p, _ in resolved:
        total_uncompressed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"  [info] uncompressed source bytes: {total_uncompressed/1e9:.2f} GB across {len(resolved)} roots")

    manifest = []
    t0 = time.time()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as zf:
        zf.writestr(f"{bundle}/LICENSES.md", LICENSES_TXT)
        zf.writestr(f"{bundle}/README.md", README_TPL.format(bundle=bundle))
        for src, arcprefix in resolved:
            for fp in src.rglob("*"):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(src)
                arcname = f"{bundle}/{arcprefix}/{rel.as_posix()}"
                zf.write(fp, arcname=arcname)
                entry = {"path": arcname, "size": fp.stat().st_size}
                if do_hash:
                    entry["sha256"] = _sha256(fp)
                manifest.append(entry)
        zf.writestr(f"{bundle}/MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"[done] {out}  size={out.stat().st_size/1e9:.2f} GB  in {time.time()-t0:.0f}s")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="kaggle_uploads", help="Output dir for .zip bundles")
    p.add_argument("--only",  nargs="*", default=None,  help="Build only these bundle names")
    p.add_argument("--hash",  action="store_true",       help="Compute sha256 per file (slow)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = (PROJECT / args.out).resolve()
    print(f"[info] project = {PROJECT}")
    print(f"[info] out     = {out_dir}")

    for bundle, srcs in BUNDLES.items():
        if args.only and bundle not in args.only:
            continue
        print(f"\n[bundle] {bundle}")
        for raw, arc in srcs:
            p_resolved = _resolve_glob_path((PROJECT / raw).resolve())
            status = "OK" if (p_resolved and p_resolved.exists()) else "MISSING"
            print(f"  [{status}] {arc}  <-  {raw}  =>  {p_resolved}")
        if args.dry_run:
            continue
        _zip_one(bundle, srcs, out_dir, do_hash=args.hash)

    if not args.dry_run:
        print("\nUpload to Kaggle (Web UI):")
        print("  https://www.kaggle.com/datasets/new  ->  upload .zip  ->  set slug + Public")
        print("Or via CLI:")
        print("  kaggle datasets create -p kaggle_uploads/voiceai-tier1-vi/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
