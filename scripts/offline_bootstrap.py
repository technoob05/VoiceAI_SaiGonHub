"""Offline bootstrap and production preflight for the VoiceAI cascade.

The script intentionally uses only the Python standard library.  It can run
before the ML environment exists and never downloads models or packages.

Examples:
    python scripts/offline_bootstrap.py preflight
    python scripts/offline_bootstrap.py preflight --quick --json
    python scripts/offline_bootstrap.py install --wheelhouse D:\\voiceai-wheels --dry-run
    python scripts/offline_bootstrap.py install --wheelhouse D:\\voiceai-wheels
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import venv
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "device" / "models.manifest.json"
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
DEFAULT_VENV = PROJECT_ROOT / ".venv"
SUPPORTED_MANIFEST_VERSION = 1
HASH_CHUNK_BYTES = 1 << 20
INCOMPLETE_SUFFIXES = (".incomplete", ".partial", ".part", ".download")

SAFE_RUNTIME_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VOICEAI_TTS_AUTO_DOWNLOAD": "0",
    "VOICEAI_CORRECTOR_LOCAL_ONLY": "1",
    "VOICEAI_DENOISE": "0",
    "VOICEAI_DENOISE_VAD": "0",
    "VOICEAI_DENOISE_GTCRN": "0",
    "VOICEAI_CACHE": "0",
    "VOICEAI_TM": "0",
    "VOICEAI_TTS_BACKEND": "piper",
    "VOICEAI_STREAMING_TTS": "1",
    "VOICEAI_LOG_CONTENT": "0",
}

SIX_DIRECTION_COMPONENTS = {
    "Vietnamese ASR": "data/raw/models_ct2/phowhisper-small-int8/",
    "multilingual ASR": "data/raw/models_ct2/whisper-small-multilingual/",
    "MT engine": "data/raw/models_ct2/nllb-200-distilled-600M-int8/",
    "MT tokenizer": "data/raw/hf_cache/hub/models--facebook--nllb-200-distilled-600M/",
    "Vietnamese corrector": "data/raw/models_ct2/vietnamese-correction-v2-int8/",
    "Vietnamese TTS": "data/raw/piper-models/vi_VN-vais1000-medium/",
    "English TTS": "data/raw/piper-models/en_US-lessac-medium/",
    "Mandarin TTS": "data/raw/piper-models/vits-melo-tts-zh_en/",
    "Korean TTS": "data/raw/piper-models/vits-mimic3-ko_KO-kss_low/",
    "ONNX frontend": "data/raw/onnx-frontend/",
}


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class CheckReport:
    findings: tuple[Finding, ...]
    checked_assets: int = 0
    checked_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not any(item.level == "error" for item in self.findings)


class PreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_asset_path(root: Path, raw: object) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw:
        return None, "asset path must be a non-empty string"
    if "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        return None, f"asset path must be repository-relative POSIX form: {raw!r}"
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        return None, f"asset path escapes or ambiguously addresses the repository: {raw!r}"
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None, f"asset path resolves outside the repository: {raw!r}"
    return candidate, None


def load_manifest(manifest_path: Path, root: Path) -> tuple[dict | None, list[Finding]]:
    findings: list[Finding] = []
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [Finding("error", "manifest_missing", f"Manifest not found: {manifest_path}")]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [Finding("error", "manifest_unreadable", f"Cannot parse {manifest_path}: {exc}")]

    if not isinstance(document, dict):
        return None, [Finding("error", "manifest_schema", "Manifest root must be a JSON object")]
    if document.get("manifest_version") != SUPPORTED_MANIFEST_VERSION:
        findings.append(Finding(
            "error", "manifest_version",
            f"Unsupported manifest_version={document.get('manifest_version')!r}; "
            f"expected {SUPPORTED_MANIFEST_VERSION}",
        ))
    assets = document.get("assets")
    if not isinstance(assets, list):
        findings.append(Finding("error", "manifest_schema", "Manifest assets must be a JSON array"))
        return document, findings
    if document.get("asset_count") != len(assets):
        findings.append(Finding(
            "error", "manifest_count",
            f"asset_count={document.get('asset_count')!r} does not match {len(assets)} rows",
        ))

    seen: set[str] = set()
    declared_total = 0
    for index, asset in enumerate(assets):
        label = f"assets[{index}]"
        if not isinstance(asset, dict):
            findings.append(Finding("error", "manifest_schema", f"{label} must be an object"))
            continue
        raw_path = asset.get("path")
        _, path_error = _safe_asset_path(root, raw_path)
        if path_error:
            findings.append(Finding("error", "manifest_path", f"{label}: {path_error}"))
        elif raw_path in seen:
            findings.append(Finding("error", "manifest_duplicate", f"Duplicate asset path: {raw_path}"))
        else:
            seen.add(raw_path)
        if not isinstance(asset.get("stage"), str) or not asset.get("stage"):
            findings.append(Finding("error", "manifest_schema", f"{label}.stage must be non-empty"))
        if not isinstance(asset.get("required", True), bool):
            findings.append(Finding("error", "manifest_schema", f"{label}.required must be boolean"))
        if asset.get("missing_dir") is True:
            continue
        size = asset.get("bytes")
        digest = asset.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            findings.append(Finding("error", "manifest_schema", f"{label}.bytes must be >= 0"))
        else:
            declared_total += size
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            findings.append(Finding("error", "manifest_schema", f"{label}.sha256 must be lowercase SHA-256"))
    if document.get("total_bytes") != declared_total:
        findings.append(Finding(
            "error", "manifest_total",
            f"total_bytes={document.get('total_bytes')!r} does not match {declared_total}",
        ))
    return document, findings


def verify_models(
    manifest_path: Path = DEFAULT_MANIFEST,
    root: Path = PROJECT_ROOT,
    *,
    quick: bool = False,
    profile: str = "six-direction",
) -> CheckReport:
    document, findings = load_manifest(manifest_path, root)
    if document is None or any(item.level == "error" for item in findings):
        return CheckReport(tuple(findings))

    declared_paths = {
        asset.get("path", "") for asset in document["assets"] if isinstance(asset, dict)
    }
    if profile == "six-direction":
        for component, prefix in SIX_DIRECTION_COMPONENTS.items():
            if not any(path.startswith(prefix) for path in declared_paths):
                findings.append(Finding(
                    "error", "profile_component_missing",
                    f"Six-direction manifest does not declare {component} under {prefix}",
                ))

    checked_assets = 0
    checked_bytes = 0
    asset_directories: set[Path] = set()
    for asset in document["assets"]:
        path, path_error = _safe_asset_path(root, asset.get("path"))
        if path_error or path is None:
            continue
        required = profile == "six-direction" or asset.get("required", True)
        level = "error" if required else "warning"
        raw_path = asset["path"]
        asset_directories.add(path.parent)
        if asset.get("missing_dir") is True:
            findings.append(Finding(level, "model_tree_missing", f"Model tree absent: {raw_path}"))
            continue
        if not path.is_file():
            findings.append(Finding(level, "model_missing", f"Model asset missing: {raw_path}"))
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            findings.append(Finding(level, "model_unreadable", f"Cannot stat {raw_path}: {exc}"))
            continue
        if size != asset["bytes"]:
            findings.append(Finding(
                level, "model_size",
                f"Model size mismatch for {raw_path}: found {size}, expected {asset['bytes']}",
            ))
            continue
        if not quick:
            try:
                digest = _sha256(path)
            except OSError as exc:
                findings.append(Finding(level, "model_unreadable", f"Cannot hash {raw_path}: {exc}"))
                continue
            if digest != asset["sha256"]:
                findings.append(Finding(
                    level, "model_hash",
                    f"SHA-256 mismatch for {raw_path}: found {digest[:16]}..., "
                    f"expected {asset['sha256'][:16]}...",
                ))
                continue
        checked_assets += 1
        checked_bytes += size

    for directory in sorted(asset_directories):
        if not directory.is_dir():
            continue
        try:
            siblings = directory.iterdir()
            for path in siblings:
                if path.is_file() and path.name.lower().endswith(INCOMPLETE_SUFFIXES):
                    findings.append(Finding(
                        "error", "incomplete_download",
                        f"Incomplete model/download artifact present: {path.relative_to(root).as_posix()}",
                    ))
        except OSError as exc:
            findings.append(Finding(
                "error", "model_tree_unreadable",
                f"Cannot scan model directory {directory}: {exc}",
            ))
    return CheckReport(tuple(findings), checked_assets, checked_bytes)


def parse_requirements(path: Path) -> tuple[dict[str, str], list[Finding]]:
    pins: dict[str, str] = {}
    findings: list[Finding] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [Finding("error", "requirements_unreadable", f"Cannot read {path}: {exc}")]
    for number, original in enumerate(lines, 1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)", line)
        if match is None:
            findings.append(Finding(
                "error", "requirement_unpinned",
                f"{path.name}:{number} is not an exact name==version pin: {line!r}",
            ))
            continue
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in pins and pins[normalized] != version:
            findings.append(Finding(
                "error", "requirement_duplicate",
                f"Conflicting pins for {name}: {pins[normalized]} and {version}",
            ))
        pins[normalized] = version
    if not pins:
        findings.append(Finding("error", "requirements_empty", f"No pinned requirements in {path}"))
    return pins, findings


def verify_dependencies(requirements: Path = DEFAULT_REQUIREMENTS) -> CheckReport:
    pins, findings = parse_requirements(requirements)
    for name, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            findings.append(Finding("error", "dependency_missing", f"Missing package: {name}=={expected}"))
            continue
        if actual != expected:
            findings.append(Finding(
                "error", "dependency_version",
                f"Version mismatch: {name}=={actual}, expected {expected}",
            ))
    return CheckReport(tuple(findings))


def verify_offline_policy(environ: dict[str, str] | None = None) -> CheckReport:
    env = os.environ if environ is None else environ
    findings: list[Finding] = []
    for name, expected in SAFE_RUNTIME_ENV.items():
        actual = env.get(name)
        if actual != expected:
            findings.append(Finding(
                "error", "offline_policy",
                f"{name} must be {expected!r} for production (found {actual!r})",
            ))
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    configured = [name for name in proxy_names if env.get(name)]
    if configured:
        findings.append(Finding(
            "warning", "proxy_configured",
            "Network proxy variables remain configured: " + ", ".join(configured),
        ))
    return CheckReport(tuple(findings))


def combine(reports: Iterable[CheckReport]) -> CheckReport:
    reports = tuple(reports)
    return CheckReport(
        tuple(item for report in reports for item in report.findings),
        sum(report.checked_assets for report in reports),
        sum(report.checked_bytes for report in reports),
    )


def pip_install_command(
    python: Path | str,
    wheelhouse: Path,
    requirements: Path,
    *,
    dry_run: bool,
) -> list[str]:
    command = [
        str(python), "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--no-index", "--find-links", str(wheelhouse.resolve()),
        "-r", str(requirements.resolve()),
    ]
    if dry_run:
        command.insert(4, "--dry-run")
    return command


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def validate_install_inputs(wheelhouse: Path, requirements: Path, venv_dir: Path) -> None:
    if not wheelhouse.is_dir():
        raise PreflightError(f"Local wheelhouse does not exist: {wheelhouse}")
    pins, findings = parse_requirements(requirements)
    del pins
    errors = [item.message for item in findings if item.level == "error"]
    if errors:
        raise PreflightError("Invalid requirements:\n  " + "\n  ".join(errors))
    if venv_dir.exists() and not (venv_dir / "pyvenv.cfg").is_file():
        raise PreflightError(
            f"Refusing to use existing non-venv path: {venv_dir}. Choose another --venv."
        )


def install_offline(
    wheelhouse: Path,
    requirements: Path,
    venv_dir: Path,
    *,
    dry_run: bool,
    manifest: Path = DEFAULT_MANIFEST,
    root: Path = PROJECT_ROOT,
    profile: str = "six-direction",
    quick: bool = False,
) -> int:
    validate_install_inputs(wheelhouse, requirements, venv_dir)
    python = Path(sys.executable) if dry_run else _venv_python(venv_dir)
    if not dry_run and not python.is_file():
        venv.EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
    command = pip_install_command(python, wheelhouse, requirements, dry_run=dry_run)
    environment = os.environ.copy()
    environment.update(SAFE_RUNTIME_ENV)
    environment.update({
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    })
    print("[bootstrap] " + subprocess.list2cmdline(command))
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        print("[bootstrap] Offline dependency resolution/install failed; no network fallback was used.", file=sys.stderr)
        return completed.returncode
    if dry_run:
        print("[bootstrap] Dry-run resolved entirely from the local wheelhouse; no files changed.")
        return 0
    preflight = [
        str(python), str(Path(__file__).resolve()), "preflight",
        "--manifest", str(manifest.resolve()),
        "--requirements", str(requirements.resolve()),
        "--root", str(root.resolve()),
        "--profile", profile,
    ]
    if quick:
        preflight.append("--quick")
    print("[bootstrap] " + subprocess.list2cmdline(preflight))
    completed = subprocess.run(preflight, env=environment, check=False)
    if completed.returncode:
        print("[bootstrap] Environment installed, but the production preflight failed.", file=sys.stderr)
    return completed.returncode


def _render(report: CheckReport, *, json_output: bool, mode: str) -> None:
    if json_output:
        print(json.dumps({
            "ok": report.ok,
            "mode": mode,
            "checked_assets": report.checked_assets,
            "checked_bytes": report.checked_bytes,
            "findings": [asdict(item) for item in report.findings],
        }, indent=2))
        return
    for item in report.findings:
        print(f"[{item.level.upper():7}] {item.code}: {item.message}")
    status = "READY" if report.ok else "NOT READY"
    print(
        f"[preflight] {status}; {report.checked_assets} assets, "
        f"{report.checked_bytes / 2**20:.1f} MiB verified ({mode})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="run the production readiness gate")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    preflight.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    preflight.add_argument("--root", type=Path, default=PROJECT_ROOT)
    preflight.add_argument("--profile", choices=("six-direction", "core", "manifest"), default="six-direction")
    preflight.add_argument("--quick", action="store_true", help="size-only model check; full SHA-256 is the default")
    preflight.add_argument("--skip-dependencies", action="store_true")
    preflight.add_argument("--skip-env", action="store_true")
    preflight.add_argument("--json", action="store_true")

    models = subparsers.add_parser("verify-models", help="verify model manifest and artifacts")
    models.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    models.add_argument("--root", type=Path, default=PROJECT_ROOT)
    models.add_argument("--profile", choices=("six-direction", "core", "manifest"), default="six-direction")
    models.add_argument("--quick", action="store_true")
    models.add_argument("--json", action="store_true")

    policy = subparsers.add_parser("check-policy", help="verify fail-closed runtime environment")
    policy.add_argument("--json", action="store_true")

    install = subparsers.add_parser("install", help="create/update an offline virtual environment")
    install.add_argument("--wheelhouse", type=Path, required=True)
    install.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    install.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    install.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    install.add_argument("--root", type=Path, default=PROJECT_ROOT)
    install.add_argument("--profile", choices=("six-direction", "core", "manifest"), default="six-direction")
    install.add_argument("--quick", action="store_true", help="use size-only model verification after install")
    install.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        try:
            return install_offline(
                args.wheelhouse.resolve(), args.requirements.resolve(), args.venv.resolve(),
                dry_run=args.dry_run,
                manifest=args.manifest.resolve(), root=args.root.resolve(),
                profile=args.profile, quick=args.quick,
            )
        except PreflightError as exc:
            print(f"[bootstrap] {exc}", file=sys.stderr)
            return 2

    if args.command == "check-policy":
        report = verify_offline_policy()
        _render(report, json_output=args.json, mode="policy")
        return 0 if report.ok else 2

    model_report = verify_models(
        args.manifest.resolve(), args.root.resolve(), quick=args.quick, profile=args.profile,
    )
    if args.command == "verify-models":
        _render(model_report, json_output=args.json, mode="size" if args.quick else "sha256")
        return 0 if model_report.ok else 2

    reports = [model_report]
    if not args.skip_dependencies:
        reports.append(verify_dependencies(args.requirements.resolve()))
    if not args.skip_env:
        reports.append(verify_offline_policy())
    report = combine(reports)
    _render(report, json_output=args.json, mode="size" if args.quick else "sha256")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
