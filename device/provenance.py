"""provenance.py — model manifest: generate SHA-256 for every artifact the
pipeline actually loads, and verify them before the supervisor starts.

Why this exists (pattern taken from references/omniglass/EDGE_DEPLOYMENT.md):
a demo device is copied around on USB sticks and re-flashed. A truncated
`model.bin` does not fail loudly at load time — CTranslate2 will often raise
something obscure deep inside the first inference, i.e. in front of the judges.
Hashing the frozen set at startup turns that into a named failure before any
model is loaded.

The asset list is derived from the code, not guessed:
  * ASR         prototype/pipeline_fast.py  FastPipelineConfig.asr_ct2_dir
  * MT          prototype/pipeline_fast.py  FastPipelineConfig.mt_ct2_dir
  * MT tokenizer  pipeline_fast._ensure_mt() -> HF cache snapshot of
                  facebook/nllb-200-distilled-600M (local_files_only=True)
  * corrector   prototype/transcript_corrector.py DEFAULT_CT2_MODEL
  * TTS voices  prototype/local_tts_piper.py DEFAULT_MODELS_ROOT
  * VAD/denoise prototype/edge_frontend.py MODELS_ROOT (silero_vad, gtcrn_simple)

Usage
    python device/provenance.py generate            # writes device/models.manifest.json
    python device/provenance.py verify              # full SHA-256, exit 0/5
    python device/provenance.py verify --quick      # size-only (demo day, seconds)

Exit codes:  0 ok | 5 mismatch/missing | 6 no manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parent
PROJECT = DEVICE_DIR.parent
# VOICEAI_MANIFEST overrides the path (used to test the mismatch path without
# touching the real manifest).
MANIFEST = Path(os.environ["VOICEAI_MANIFEST"]) if os.environ.get("VOICEAI_MANIFEST") \
    else DEVICE_DIR / "models.manifest.json"

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# --- what the pipeline loads -------------------------------------------------
# (stage, directory relative to PROJECT, filename globs, required)
ASSET_GROUPS = [
    ("asr", "data/raw/models_ct2/phowhisper-small-int8", ["*"], True),
    # Not required: the multilingual ASR is an opt-in second resident model
    # (live_translate --multilingual-asr). A Vietnamese-only device never loads it,
    # so its absence must not fail startup - but if it IS on disk it must verify.
    ("asr_multilingual", "data/raw/models_ct2/whisper-small-multilingual", ["*"], False),
    ("mt", "data/raw/models_ct2/nllb-200-distilled-600M-int8", ["*"], True),
    ("corrector", "data/raw/models_ct2/vietnamese-correction-v2-int8", ["*"], False),
    ("tts", "data/raw/piper-models/vi_VN-vais1000-medium",
     ["*.onnx", "*.onnx.json", "tokens.txt"], True),
    ("tts", "data/raw/piper-models/en_US-lessac-medium",
     ["*.onnx", "*.onnx.json", "tokens.txt"], True),
    ("tts", "data/raw/piper-models/vits-melo-tts-zh_en",
     ["model.onnx", "tokens.txt", "lexicon.txt"], False),
    ("tts", "data/raw/piper-models/vits-mimic3-ko_KO-kss_low",
     ["*.onnx", "*.onnx.json", "tokens.txt"], False),
    ("frontend", "data/raw/onnx-frontend", ["silero_vad.onnx", "gtcrn_simple.onnx"], True),
]

# HF-cache snapshot dirs (symlink farm) — resolved to real blobs when hashed.
HF_SNAPSHOT_GROUPS = [
    ("mt_tokenizer", "data/raw/hf_cache/hub/models--facebook--nllb-200-distilled-600M", True),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _snapshot_files(repo_dir: Path) -> list[Path]:
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return []
    out: list[Path] = []
    for rev in sorted(snaps.iterdir()):
        for p in sorted(rev.rglob("*")):
            if p.is_file() or p.is_symlink():
                try:
                    if p.resolve().is_file():
                        out.append(p)
                except OSError:
                    pass
    return out


def collect() -> list[dict]:
    rows: list[dict] = []
    for stage, rel, globs, required in ASSET_GROUPS:
        d = PROJECT / rel
        if not d.is_dir():
            rows.append({"stage": stage, "path": rel, "required": required,
                         "missing_dir": True})
            continue
        seen: set[Path] = set()
        for g in globs:
            for p in sorted(d.glob(g)):
                if p.is_file() and p not in seen and p.name != ".complete":
                    seen.add(p)
        for p in sorted(seen):
            rows.append({
                "stage": stage,
                "path": p.relative_to(PROJECT).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "required": required,
            })
    for stage, rel, required in HF_SNAPSHOT_GROUPS:
        d = PROJECT / rel
        for p in _snapshot_files(d):
            real = p.resolve()
            rows.append({
                "stage": stage,
                "path": p.relative_to(PROJECT).as_posix(),
                "bytes": real.stat().st_size,
                "sha256": sha256_file(real),
                "required": required,
            })
    return rows


def generate() -> int:
    t0 = time.perf_counter()
    rows = collect()
    total = sum(r.get("bytes", 0) for r in rows)
    doc = {
        "manifest_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_on": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "note": ("Hashes generated against the real files on the Windows dev box. "
                 "Copy this file with the models; regenerating it on the target "
                 "device defeats the purpose."),
        "total_bytes": total,
        "asset_count": len(rows),
        "assets": rows,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"[provenance] wrote {MANIFEST}")
    print(f"[provenance] {len(rows)} assets, {total/2**20:.1f} MiB, hashed in {dt:.1f}s "
          f"({total/2**20/max(dt,1e-9):.0f} MiB/s)")
    for stage in sorted({r['stage'] for r in rows}):
        n = sum(1 for r in rows if r["stage"] == stage)
        b = sum(r.get("bytes", 0) for r in rows if r["stage"] == stage)
        print(f"   {stage:<13} {n:>3} files  {b/2**20:>8.1f} MiB")
    return 0


def verify(quick: bool = False, quiet: bool = False) -> int:
    if not MANIFEST.is_file():
        print(f"[provenance] FAIL: no manifest at {MANIFEST}", file=sys.stderr)
        return 6
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    bad: list[str] = []
    warn: list[str] = []
    checked = 0
    nbytes = 0
    for a in doc["assets"]:
        p = PROJECT / a["path"]
        required = a.get("required", True)
        if a.get("missing_dir"):
            (bad if required else warn).append(f"{a['path']}: directory absent at generate time")
            continue
        if not p.is_file():
            (bad if required else warn).append(f"{a['path']}: MISSING")
            continue
        size = p.stat().st_size
        if size != a["bytes"]:
            (bad if required else warn).append(
                f"{a['path']}: SIZE {size} != {a['bytes']}")
            continue
        if not quick:
            got = sha256_file(p)
            if got != a["sha256"]:
                (bad if required else warn).append(
                    f"{a['path']}: SHA256 {got[:16]}… != {a['sha256'][:16]}…")
                continue
            nbytes += size
        checked += 1
    dt = time.perf_counter() - t0
    mode = "size-only" if quick else "sha256"
    if not quiet:
        print(f"[provenance] {mode}: {checked}/{len(doc['assets'])} assets ok "
              f"in {dt:.1f}s" + ("" if quick else f" ({nbytes/2**20/max(dt,1e-9):.0f} MiB/s)"))
    for w in warn:
        print(f"[provenance] WARN (optional asset) {w}")
    if bad:
        print("[provenance] *** MANIFEST VERIFICATION FAILED ***", file=sys.stderr)
        for b in bad:
            print(f"[provenance]   {b}", file=sys.stderr)
        print("[provenance] Refusing to declare the device ready. Re-copy the model "
              "tree from the master USB, then re-run.", file=sys.stderr)
        return 5
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["generate", "verify"])
    ap.add_argument("--quick", action="store_true",
                    help="verify: size-only, skips hashing (seconds, weaker)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.action == "generate":
        return generate()
    return verify(quick=args.quick, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
