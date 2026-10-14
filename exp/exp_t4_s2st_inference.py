"""Private Kaggle T4 benchmark for the local VoiceAI cascade.

The kernel must run with Internet disabled and two pinned Kaggle inputs:
  - daosyduyminh/voiceai-edge-models-t4
  - daosyduyminh/voiceai-saigonhub-tier-1-speech-vi-zh-ko-en

It measures warm per-utterance ASR, safe Vietnamese correction, MT, Piper TTS,
end-to-end latency, RTF, and process-wide GPU memory. Raw vectors are retained
so p50/p95 can be recomputed rather than inferred from summary statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = HERE / "benchmark_config.json"
KAGGLE_INPUT = Path("/kaggle/input")


def dataset_root(slug: str) -> Path:
    """Resolve a Kaggle dataset mount across both legacy and current layouts.

    Legacy kernels mount at /kaggle/input/<slug>; current ones mount at
    /kaggle/input/datasets/<owner>/<slug>. Neither is guaranteed, so probe both
    and fall back to a bounded scan instead of assuming one shape.
    """
    direct = KAGGLE_INPUT / slug
    if direct.is_dir():
        return direct
    owners = KAGGLE_INPUT / "datasets"
    if owners.is_dir():
        for owner in sorted(owners.iterdir()):
            candidate = owner / slug
            if candidate.is_dir():
                return candidate
    if KAGGLE_INPUT.is_dir():
        matches = sorted(
            (p for p in KAGGLE_INPUT.glob(f"*/*/{slug}") if p.is_dir()),
            key=lambda p: len(p.parts),
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Kaggle dataset not mounted: {slug}")


def kaggle_config_path() -> Path:
    try:
        return dataset_root("voiceai-t4-benchmark-config") / "benchmark_config.json"
    except FileNotFoundError:
        return Path("/nonexistent/benchmark_config.json")


KAGGLE_CONFIG_PATH = kaggle_config_path()
DEFAULT_CONFIG = {
    "condition": "smoke-t4-int8-v1",
    "samples": 2,
    "warmup_samples": 1,
    "target": "en",
    "required_gpu_substring": "T4",
    "vram_limit_gib": 10.0,
    "asr_compute_type": "int8_float16",
    "text_compute_type": "int8_float16",
    "seed": 20260817,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    config_path = KAGGLE_CONFIG_PATH if KAGGLE_CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    if config_path.exists():
        cfg.update(json.loads(config_path.read_text(encoding="utf-8")))
    cfg["resolved_config_path"] = str(config_path)
    return cfg


def unwrap_same_name(path: Path) -> Path:
    """Descend through Kaggle's ZIP auto-extract wrappers (``wheels/wheels``)."""
    while True:
        nested = path / path.name
        if not nested.is_dir():
            return path
        path = nested


def payload_dir(root: Path, name: str) -> Path:
    """Find a payload directory regardless of Kaggle's ZIP auto-extract nesting."""
    direct = root / name
    if direct.is_dir():
        return unwrap_same_name(direct)
    matches = sorted(
        (path for path in root.rglob(name) if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    if matches:
        return unwrap_same_name(matches[0])
    raise FileNotFoundError(
        f"Missing payload directory {name!r} under {root} "
        f"(children: {[p.name for p in root.iterdir()] if root.is_dir() else 'n/a'})"
    )


def select_wheels(wheel_files: list[Path]) -> list[Path]:
    """Keep one wheel per distribution, matching the running interpreter.

    The first model bundle shipped cp311 wheels; the Kaggle image has since
    moved to CPython 3.12, so pick by ABI tag instead of by upload order.
    """
    abi_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    chosen: dict[str, Path] = {}
    for path in sorted(wheel_files):
        parts = path.name.split("-")
        if len(parts) < 5:
            continue
        dist, python_tag = parts[0].lower().replace("_", "-"), parts[2]
        pure = python_tag.startswith("py3") or python_tag.startswith("py2.py3")
        if not (pure or abi_tag in python_tag.split(".")):
            continue
        if dist not in chosen or not pure:
            chosen[dist] = path
    return [chosen[key] for key in sorted(chosen)]


def install_offline_wheels(roots: list[Path]) -> None:
    found: list[Path] = []
    for root in roots:
        found.extend(path for path in root.rglob("*.whl") if path.is_file())
    wheels = select_wheels(found)
    if not wheels:
        raise FileNotFoundError(
            f"No wheels for cp{sys.version_info.major}{sys.version_info.minor} "
            f"among {[p.name for p in found]}"
        )
    print(f"[wheels] installing {[p.name for p in wheels]}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
         "--no-deps", *map(str, wheels)],
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GpuMonitor:
    def __init__(self, index: int = 0):
        self.index = index
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception as exc:
            print(f"[warn] NVML monitor unavailable: {exc}")
        return self

    def _poll(self):
        while not self._stop.is_set():
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            self.samples.append({
                "time": time.time(),
                "used_mib": mem.used / (1024 ** 2),
                "total_mib": mem.total / (1024 ** 2),
                "gpu_util_pct": util.gpu,
            })
            self._stop.wait(0.1)

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        return {
            "available": True,
            "peak_used_mib": max(row["used_mib"] for row in self.samples),
            "total_mib": max(row["total_mib"] for row in self.samples),
            "peak_gpu_util_pct": max(row["gpu_util_pct"] for row in self.samples),
            "n_samples": len(self.samples),
        }


def pct(values: list[float], q: float) -> float:
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict]) -> dict:
    metrics = ("asr_ms", "corrector_ms", "mt_ms", "tts_ms", "total_ms", "rtf")
    result = {}
    for key in metrics:
        values = [float(row[key]) for row in rows]
        result[key] = {
            "mean": statistics.fmean(values),
            "p50": pct(values, 50),
            "p95": pct(values, 95),
            "min": min(values),
            "max": max(values),
        }
    result["rtf_pass_rate"] = sum(row["rtf"] < 1.0 for row in rows) / max(len(rows), 1)
    result["under_2s_rate"] = sum(row["total_ms"] < 2000 for row in rows) / max(len(rows), 1)
    return result


def main() -> int:
    cfg = load_config()
    input_root = Path("/kaggle/input")
    output_root = Path("/kaggle/working")
    if not input_root.exists():
        raise RuntimeError("This benchmark is intended for a Kaggle kernel")

    model_root = dataset_root("voiceai-edge-models-t4")
    speech_root = payload_dir(
        dataset_root("voiceai-saigonhub-tier-1-speech-vi-zh-ko-en"), "vivos"
    )
    print(f"[paths] model_root={model_root}")
    print(f"[paths] speech_root={speech_root}")
    wheel_roots = [model_root]
    try:
        wheel_roots.insert(0, dataset_root("voiceai-t4-wheels-py312"))
    except FileNotFoundError:
        pass
    install_offline_wheels(wheel_roots)
    sys.path.insert(0, str(HERE))
    try:
        sys.path.insert(0, str(dataset_root("voiceai-t4-srclib")))
    except FileNotFoundError:
        pass

    import ctranslate2
    import numpy as np
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from transformers import BartphoTokenizer, NllbTokenizerFast
    from local_tts_piper import PiperTTS
    from transcript_corrector import CorrectorConfig, TranscriptCorrector

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not gpu_names or cfg["required_gpu_substring"].lower() not in gpu_names[0].lower():
        failure = {"status": "failed_gpu_gate", "config": cfg, "gpu_names": gpu_names}
        (output_root / "voiceai_t4_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"T4 required, got {gpu_names}")

    models_dir = payload_dir(model_root, "models_ct2")
    piper_dir = payload_dir(model_root, "piper-models")
    asr_dir = models_dir / "phowhisper-small-int8"
    mt_dir = models_dir / "nllb-200-distilled-600M-int8"
    corr_dir = models_dir / "vietnamese-correction-v2-int8"
    os.environ["VOICEAI_PIPER_MODELS_DIR"] = str(piper_dir)

    tts_model = max(
        (piper_dir / "en_US-lessac-medium").glob("*.onnx"),
        key=lambda path: path.stat().st_size,
    )
    model_files = {
        "asr": asr_dir / "model.bin",
        "mt": mt_dir / "model.bin",
        "corrector": corr_dir / "model.bin",
        "tts": tts_model,
    }
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} artifact: {path}")

    load_ms = {}
    started = time.perf_counter()
    asr = WhisperModel(
        str(asr_dir), device="cuda", device_index=0,
        compute_type=cfg["asr_compute_type"], cpu_threads=2, num_workers=1,
    )
    load_ms["asr"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    corr_translator = ctranslate2.Translator(
        str(corr_dir), device="cuda", device_index=0,
        compute_type=cfg["text_compute_type"], inter_threads=1,
    )
    corr_tok = BartphoTokenizer.from_pretrained(str(corr_dir), local_files_only=True)

    def correct_generate(text: str) -> str:
        ids = corr_tok(text, add_special_tokens=True)["input_ids"]
        tokens = corr_tok.convert_ids_to_tokens(ids)
        result = corr_translator.translate_batch(
            [tokens], beam_size=1,
            max_decoding_length=min(256, max(len(tokens) + 16, 24)),
            repetition_penalty=1.05, no_repeat_ngram_size=4,
        )[0]
        out_ids = corr_tok.convert_tokens_to_ids(result.hypotheses[0])
        return corr_tok.decode(out_ids, skip_special_tokens=True)

    corrector = TranscriptCorrector(
        CorrectorConfig(dynamic_int8=False, ct2_model=""), generator=correct_generate
    )
    load_ms["corrector"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    mt = ctranslate2.Translator(
        str(mt_dir), device="cuda", device_index=0,
        compute_type=cfg["text_compute_type"], inter_threads=1,
    )
    mt_tok = NllbTokenizerFast.from_pretrained(str(mt_dir), local_files_only=True)
    mt_tok.src_lang = "vie_Latn"
    load_ms["mt"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    tts = PiperTTS(num_threads=2, auto_download=False)
    tts.synthesize("Ready.", cfg["target"])
    load_ms["tts"] = (time.perf_counter() - started) * 1000

    def translate(text: str) -> str:
        source_tokens = mt_tok.convert_ids_to_tokens(mt_tok.encode(text))
        max_dec = min(256, max(16, len(source_tokens) * 2 + 4))
        result = mt.translate_batch(
            [source_tokens], target_prefix=[["eng_Latn"]], beam_size=1,
            max_decoding_length=max_dec,
        )[0]
        tokens = result.hypotheses[0][1:]
        return mt_tok.decode(mt_tok.convert_tokens_to_ids(tokens), skip_special_tokens=True).strip()

    prompt_map = {}
    for line in (speech_root / "test" / "prompts.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            prompt_map[parts[0]] = parts[1]
    wavs = sorted((speech_root / "test" / "waves").rglob("*.wav"))
    needed = int(cfg["warmup_samples"]) + int(cfg["samples"])
    if len(wavs) < needed:
        raise RuntimeError(f"Need {needed} wavs, found {len(wavs)}")

    def run_one(path: Path) -> dict:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        segments, _ = asr.transcribe(
            audio, language="vi", task="transcribe", beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
            without_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            no_repeat_ngram_size=3, repetition_penalty=1.2,
            compression_ratio_threshold=1.8,
        )
        raw_vi = " ".join(segment.text.strip() for segment in segments).strip()
        asr_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        correction = corrector.correct(raw_vi)
        vi = correction.text
        corrector_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        en = translate(vi)
        mt_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        out_audio, out_sr = tts.synthesize(en, cfg["target"])
        tts_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000
        return {
            "id": path.stem,
            "duration_s": duration,
            "reference_vi": prompt_map.get(path.stem),
            "raw_vi": raw_vi,
            "corrected_vi": vi,
            "correction_accepted": correction.accepted,
            "correction_reason": correction.reason,
            "target_text": en,
            "output_audio_s": len(out_audio) / max(out_sr, 1),
            "asr_ms": asr_ms,
            "corrector_ms": corrector_ms,
            "mt_ms": mt_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
            "rtf": total_ms / max(duration * 1000, 1),
        }

    with GpuMonitor(0) as monitor:
        warmups = [run_one(path) for path in wavs[: int(cfg["warmup_samples"])]]
        rows = [run_one(path) for path in wavs[int(cfg["warmup_samples"]):needed]]
    gpu_usage = monitor.summary()

    condition_payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    report = {
        "status": "complete",
        "condition_id": hashlib.sha256(condition_payload.encode()).hexdigest()[:16],
        "config": cfg,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_names": gpu_names,
            "gpu_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ctranslate2": ctranslate2.__version__,
            "ctranslate2_cuda_devices": ctranslate2.get_cuda_device_count(),
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "model_sha256": {name: sha256(path) for name, path in model_files.items()},
        "model_load_ms": load_ms,
        "gpu_usage": gpu_usage,
        "warmup_rows": warmups,
        "rows": rows,
        "summary": summarize(rows),
        "notes": [
            "T4 is a development benchmark, not the competition deployment device.",
            "Piper TTS runs on CPU; ASR, corrector, and MT run on CUDA device 0.",
            "total_ms is conservative batch latency to complete synthesized audio.",
        ],
    }

    out_name = f"voiceai_{cfg['condition']}.json"
    (output_root / out_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / f"voiceai_{cfg['condition']}_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    peak_mib = gpu_usage.get("peak_used_mib")
    if peak_mib is None:
        raise RuntimeError("GPU memory monitor unavailable; benchmark is incomplete")
    if peak_mib > float(cfg["vram_limit_gib"]) * 1024:
        raise RuntimeError(f"VRAM gate failed: {peak_mib:.1f} MiB")
    print(json.dumps(report["summary"], indent=2))
    print(f"[result] {out_name} peak_vram={peak_mib:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
