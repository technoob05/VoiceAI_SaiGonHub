"""Local offline neural TTS — Piper / MeloTTS (VITS) on-device via sherpa-onnx.

Python port of my-translator's `src-tauri/src/commands/local_tts.rs`
(references/my-translator), adapted for the S2ST research pipeline:

  * 100% offline: espeak-ng-data (Piper) or lexicon+dict (MeloTTS) ship inside
    each model archive — no remote phonemizer, no cloud call. This is the only
    TTS path that satisfies OneVoice's on-device constraint. (The repo's
    "Edge TTS" is a Microsoft *cloud* WebSocket service — not usable here.)
  * Integrity-gated model download (SHA-256, atomic install) — same reproducibility
    contract the paper needs when it reports which voice produced which audio.
  * Engine LRU cache so bilingual switching does not reload the 1-3 s model each turn.
  * Auto-detects the phonemizer layout so the SAME loader drives Piper (espeak),
    MeloTTS-zh (jieba dict + lexicon), and mimic3-ko — closing the Mandarin gap
    that MMS-TTS leaves as silence.

Contract matches `pipeline_fast.FastMTPipeline.synthesize`:
    synthesize(text, target) -> (np.ndarray float32 mono, sample_rate int)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_ROOT = PROJECT / "data" / "raw" / "piper-models"

RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
CUSTOM_VI_BASE = "https://github.com/phuc-nt/my-translator/releases/download/tts-models-vi"


class VoiceEntry:
    __slots__ = ("id", "display", "lang", "package", "base_url", "sha256", "sample_rate")

    def __init__(self, id, display, lang, package, base_url, sha256, sample_rate):
        self.id = id
        self.display = display
        self.lang = lang
        self.package = package
        self.base_url = base_url
        self.sha256 = sha256          # None => integrity check skipped (warn)
        self.sample_rate = sample_rate

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.package}.tar.bz2"


# SHA-256 for vi/en pinned from my-translator's verified Rust CATALOG.
# zh/ko point at upstream sherpa-onnx releases; sha256=None (no pinned hash yet).
CATALOG: list[VoiceEntry] = [
    VoiceEntry("vi_VN-vais1000-medium", "Tiếng Việt — VAIS1000 (medium)", "vi",
               "vits-piper-vi_VN-vais1000-medium", RELEASE_BASE,
               "fa1367710767d36ed5cf13b4a449e20c35ffd12791c2e47c2e64142bfa55551a", 22_050),
    VoiceEntry("vi_VN-25hours_single-low", "Tiếng Việt — 25 hours (low)", "vi",
               "vits-piper-vi_VN-25hours_single-low", RELEASE_BASE,
               "8aa8bbe88a1cb26ef4f33de56fced1720e72ec00491d23a3551de25a53c75149", 16_000),
    VoiceEntry("en_US-lessac-medium", "English (US) — Lessac (medium)", "en",
               "vits-piper-en_US-lessac-medium", RELEASE_BASE,
               "9e3febfacf0abf4270172d2958bcec246032b7e88efc2720840cc80c93de334e", 22_050),
    VoiceEntry("en_US-ryan-medium", "English (US) — Ryan (medium)", "en",
               "vits-piper-en_US-ryan-medium", RELEASE_BASE,
               "c546af78b6395b4e7c4ce1ed899438b64426a362f5d4ec5fecd090ded9ad7505", 22_050),
    # Mandarin — MeloTTS zh+en (jieba dict + lexicon inside archive). Closes the
    # zh gap that MMS-TTS leaves as silence.
    VoiceEntry("vits-melo-tts-zh_en", "中文/English — MeloTTS (zh_en)", "zh",
               "vits-melo-tts-zh_en", RELEASE_BASE, None, 44_100),
    # Korean — mimic3 KSS VITS.
    VoiceEntry("vits-mimic3-ko_KO-kss_low", "한국어 — KSS (mimic3 low)", "ko",
               "vits-mimic3-ko_KO-kss_low", RELEASE_BASE, None, 22_050),
    # --- Faster Mandarin alternatives -------------------------------------
    # MeloTTS is the quality voice but costs ~1185 ms per second of synthesized
    # audio against ~157 ms/s for Piper en (measured, see exp/bench_tts_mandarin.py):
    # 2x of that is the 44.1 kHz output rate, the rest is a 170 MB net against 63 MB.
    # These two are drop-in VITS voices (same loader path) for when latency wins
    # over fidelity. Swap with VOICEAI_PIPER_VOICE_ZH=<id> — no code change.
    # aishell3 is 8 kHz and Mandarin-only: it drops embedded Latin words, so it is
    # NOT a like-for-like replacement for zh_en code-switched industrial text.
    VoiceEntry("vits-icefall-zh-aishell3", "中文 — icefall AISHELL-3 (8 kHz, fast)", "zh",
               "vits-icefall-zh-aishell3", RELEASE_BASE,
               "ab468db3a3308cdd861495e0db2f25d79418a0c00639f74944c7cdf5dd8c6ec1", 8_000),
    VoiceEntry("vits-zh-hf-fanchen-C", "中文 — fanchen-C (16 kHz)", "zh",
               "vits-zh-hf-fanchen-C", RELEASE_BASE, None, 16_000),
]

# Optional pre-quantized weights published alongside a voice. The MeloTTS archive
# ships `model.int8.onnx` as a 133-byte git-lfs pointer; the pointer names the real
# blob and its SHA-256, which is what this table records. Installed under a
# *different* filename so the stub in the archive is left untouched.
#
# Measured on x86-64 (i7-13620H, onnxruntime 1.26 CPU EP): the int8 MeloTTS graph is
# 3.5x SLOWER than fp32 (p50 10570 ms vs 2983 ms, interleaved, n=45) because dynamic
# QUInt8 falls back to un-fused MatMulInteger/Conv paths. Kept as a seam for the
# Snapdragon/QNN target where int8 is the native path — hence opt-in, never default.
QUANTIZED = {
    "vits-melo-tts-zh_en": (
        "https://huggingface.co/csukuangfj/vits-melo-tts-zh_en/resolve/main/model.int8.onnx",
        "f085f5079e05f039b800aeb542f5253c26a303211b0c6465d0d9387977855a63",
        "model.quant.onnx"),
}

# Default voice id per pipeline target language.
DEFAULT_VOICE = {
    "vi": "vi_VN-vais1000-medium",
    "en": "en_US-lessac-medium",
    "zh": "vits-melo-tts-zh_en",
    "ko": "vits-mimic3-ko_KO-kss_low",
}

MAX_EXTRACT_BYTES = 500_000_000


def catalog_entry(voice_id: str) -> VoiceEntry:
    for v in CATALOG:
        if v.id == voice_id:
            return v
    raise KeyError(f"Unknown voice id: {voice_id}")


def _ensure_safe_id(voice_id: str) -> None:
    if (not voice_id or voice_id in (".", "..") or "/" in voice_id or "\\" in voice_id
            or not all(c.isalnum() or c in "_-." for c in voice_id)):
        raise ValueError(f"Invalid voice id: {voice_id}")


def models_root() -> Path:
    configured = os.environ.get("VOICEAI_PIPER_MODELS_DIR", "").strip()
    root = Path(configured) if configured else DEFAULT_MODELS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _model_dir(root: Path, voice_id: str) -> Path:
    _ensure_safe_id(voice_id)
    return root / voice_id


def is_installed(dir_: Path) -> bool:
    return (dir_ / ".complete").exists()


def _quantized_onnx(dir_: Path) -> Optional[Path]:
    """A real (non-stub) quantized graph in this voice dir, or None."""
    seen = []
    for p in [dir_ / "model.quant.onnx", dir_ / "model.int8.onnx", *sorted(dir_.glob("*.int8.onnx"))]:
        if p in seen or not p.is_file():
            continue
        seen.append(p)
        if p.stat().st_size > 4096:     # skip the 133-byte git-lfs pointer stubs
            return p
    return None


def _resolve_model_files(dir_: Path, prefer_quantized: bool = False) -> dict:
    """Locate model files. Handles Piper (espeak-ng-data), MeloTTS (lexicon+dict),
    and mimic3 (lexicon) layouts by presence, not by hard-coded names."""
    if not is_installed(dir_):
        raise FileNotFoundError(f"Voice not installed: {dir_.name}")
    # Pick the LARGEST .onnx — some archives ship a tiny git-lfs pointer stub
    # (e.g. MeloTTS `model.int8.onnx`, 133 bytes) alongside the real `model.onnx`.
    onnx_files = [p for p in dir_.iterdir() if p.suffix == ".onnx" and p.stat().st_size > 4096]
    if not onnx_files:
        raise FileNotFoundError("No usable .onnx model file found")
    onnx = max(onnx_files, key=lambda p: p.stat().st_size)
    if prefer_quantized:
        q = _quantized_onnx(dir_)
        if q is not None:
            onnx = q
    tokens = dir_ / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError("Model is missing tokens.txt")
    files = {"model": str(onnx), "tokens": str(tokens)}
    espeak = dir_ / "espeak-ng-data"
    lexicon = dir_ / "lexicon.txt"
    dict_dir = dir_ / "dict"
    if espeak.is_dir():
        files["data_dir"] = str(espeak)
    if lexicon.exists():
        files["lexicon"] = str(lexicon)
    if dict_dir.is_dir():
        files["dict_dir"] = str(dict_dir)
    # MeloTTS text-normalization rules (numbers/dates/phones) if the archive ships them.
    fsts = [str(dir_ / n) for n in ("date.fst", "number.fst", "phone.fst")
            if (dir_ / n).exists()]
    if fsts:
        files["rule_fsts"] = ",".join(fsts)
    if "data_dir" not in files and "lexicon" not in files:
        raise FileNotFoundError("Model has neither espeak-ng-data nor lexicon.txt")
    return files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_stripping_top(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(archive, "r:bz2") as tar:
        for m in tar:
            total += max(m.size, 0)
            if total > MAX_EXTRACT_BYTES:
                raise ValueError("Archive exceeds maximum allowed size")
            parts = Path(m.name).parts[1:]  # strip leading vits-.../ component
            if not parts:
                continue
            rel = Path(*parts)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe path in archive")
            out = dest / rel
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def download_model(voice_id: str, force: bool = False) -> Path:
    """Download + SHA-256 verify + extract a catalog voice. Idempotent."""
    import urllib.request

    entry = catalog_entry(voice_id)
    root = models_root()
    final = _model_dir(root, voice_id)
    if is_installed(final) and not force:
        return final

    tmp = root / f".tmp-{voice_id}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        archive = tmp / "archive.tar.bz2"
        print(f"[piper] downloading {entry.url}")
        urllib.request.urlretrieve(entry.url, archive)
        if entry.sha256:
            got = _sha256(archive)
            if got.lower() != entry.sha256.lower():
                raise ValueError(f"SHA-256 mismatch for {voice_id}: {got}")
        else:
            print(f"[piper][warn] no pinned SHA-256 for {voice_id} — integrity check skipped")
        extract_dir = tmp / "extract"
        _extract_stripping_top(archive, extract_dir)
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        shutil.move(str(extract_dir), str(final))
        (final / ".complete").write_bytes(b"1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def download_quantized(voice_id: str, force: bool = False) -> Path:
    """Fetch the pre-quantized weights for an already-installed voice (see QUANTIZED).

    Written next to the fp32 graph under its own filename, so nothing the release
    archive shipped is overwritten. Only used when VOICEAI_TTS_QUANT is enabled.
    """
    import urllib.request

    if voice_id not in QUANTIZED:
        raise KeyError(f"No quantized weights registered for {voice_id}")
    url, sha, name = QUANTIZED[voice_id]
    dir_ = _model_dir(models_root(), voice_id)
    if not is_installed(dir_):
        raise FileNotFoundError(f"Install the voice first: {voice_id}")
    final = dir_ / name
    if final.exists() and final.stat().st_size > 4096 and not force:
        return final
    tmp = dir_ / f".tmp-{name}"
    try:
        print(f"[piper] downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        got = _sha256(tmp)
        if got.lower() != sha.lower():
            raise ValueError(f"SHA-256 mismatch for {voice_id} quantized weights: {got}")
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return final


def list_installed() -> list[str]:
    root = models_root()
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and is_installed(p) and not p.name.startswith("."))


class PiperTTS:
    """Offline VITS synthesizer with a small warm-engine LRU (default cap 2)."""

    def __init__(self, num_threads: Optional[int] = None, cache_cap: int = 2,
                 auto_download: bool = True, provider: Optional[str] = None):
        self.num_threads = num_threads or int(os.environ.get("VOICEAI_TTS_THREADS", "4"))
        self._thread_override = num_threads is not None
        # sherpa-onnx selects the ORT execution provider by name. "xnnpack" is
        # the one worth setting on the ARM target; the Windows dev wheel ships
        # only CPUExecutionProvider and logs a fallback, so it is a no-op here
        # rather than a speedup. Default "cpu" keeps the demo path unchanged.
        self.provider = (provider or os.environ.get("VOICEAI_TTS_PROVIDER", "cpu")).strip()
        self.cache_cap = cache_cap
        self.auto_download = auto_download
        self._engines: list[tuple[str, object]] = []   # MRU last

    def _voice_for(self, target: str) -> str:
        override = os.environ.get(f"VOICEAI_PIPER_VOICE_{target.upper()}", "").strip()
        if override:
            return override
        try:
            return DEFAULT_VOICE[target]
        except KeyError:
            raise KeyError(f"No local voice mapped for target '{target}'")

    def _threads_for(self, voice_id: str) -> int:
        """Thread count for one voice, because the voices are not equally costly.

        Measured on a 12-core dev box, MeloTTS Mandarin (`ms per second of output
        audio`, so the 44.1 kHz sample rate is already accounted for):

            threads  2 -> 1211    4 -> 880    6 -> 922    8 -> 786

        against 110-149 ms/audio-s for the Vietnamese, English and Korean voices
        at the default. Mandarin is ~6x the cost per unit of audio and is the only
        voice that visibly benefits from more threads, so it gets its own budget
        instead of raising the default for everyone and stealing cores from ASR.
        """
        if self._thread_override:
            return self.num_threads
        try:
            lang = catalog_entry(voice_id).lang
        except KeyError:
            return self.num_threads
        override = os.environ.get(f"VOICEAI_TTS_THREADS_{lang.upper()}", "").strip()
        if override:
            return max(1, int(override))
        if lang == "zh":
            return max(self.num_threads, min(8, os.cpu_count() or self.num_threads))
        return self.num_threads

    def _get_engine(self, voice_id: str):
        for i, (k, eng) in enumerate(self._engines):
            if k == voice_id:
                self._engines.append(self._engines.pop(i))  # touch MRU
                return eng
        eng = self._build_engine(voice_id)
        self._engines.append((voice_id, eng))
        while len(self._engines) > self.cache_cap:
            self._engines.pop(0)
        return eng

    def _build_engine(self, voice_id: str):
        import sherpa_onnx

        root = models_root()
        dir_ = _model_dir(root, voice_id)
        if not is_installed(dir_):
            if not self.auto_download:
                raise FileNotFoundError(
                    f"Voice '{voice_id}' not installed. Run: "
                    f"python local_tts_piper.py download {voice_id}")
            download_model(voice_id)
        files = _resolve_model_files(dir_)
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=files["model"],
            tokens=files["tokens"],
            data_dir=files.get("data_dir", ""),
            lexicon=files.get("lexicon", ""),
            dict_dir=files.get("dict_dir", ""),
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=vits, num_threads=self._threads_for(voice_id),
                provider=self.provider),
            rule_fsts=files.get("rule_fsts", ""),
        )
        if not config.validate():
            raise RuntimeError(f"Invalid sherpa-onnx TTS config for {voice_id}")
        print(f"[load] Piper/sherpa-onnx TTS: {voice_id} "
              f"(provider={self.provider}, threads={self._threads_for(voice_id)})")
        return sherpa_onnx.OfflineTts(config)

    def synthesize(self, text: str, target: str, speed: float = 1.0,
                   sid: int = 0) -> tuple[np.ndarray, int]:
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32), 16_000
        voice_id = self._voice_for(target)
        engine = self._get_engine(voice_id)
        audio = engine.generate(text.strip(), sid=sid, speed=speed if speed > 0 else 1.0)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


_SINGLETON: Optional[PiperTTS] = None


def get_piper() -> PiperTTS:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = PiperTTS()
    return _SINGLETON


if __name__ == "__main__":
    import argparse

    try:
        import sys as _sys
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Offline Piper/MeloTTS via sherpa-onnx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("--catalog", action="store_true")
    d = sub.add_parser("download"); d.add_argument("voice"); d.add_argument("--force", action="store_true")
    sp = sub.add_parser("speak")
    sp.add_argument("text"); sp.add_argument("--target", default="vi")
    sp.add_argument("--voice", default=None); sp.add_argument("--speed", type=float, default=1.0)
    sp.add_argument("--out", default="piper_out.wav")
    args = ap.parse_args()

    if args.cmd == "list":
        if getattr(args, "catalog", False):
            for v in CATALOG:
                print(f"  {v.id:32s} [{v.lang}] {v.display}  ({v.sample_rate} Hz)")
        else:
            print("installed:", list_installed() or "(none)")
    elif args.cmd == "download":
        p = download_model(args.voice, force=args.force)
        print("installed at", p)
    elif args.cmd == "speak":
        import soundfile as sf
        tts = PiperTTS()
        if args.voice:
            os.environ[f"VOICEAI_PIPER_VOICE_{args.target.upper()}"] = args.voice
        wav, sr = tts.synthesize(args.text, args.target, speed=args.speed)
        sf.write(args.out, wav, sr)
        print(f"wrote {args.out}  ({len(wav)/sr:.2f}s @ {sr} Hz)")
