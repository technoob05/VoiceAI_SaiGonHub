from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import offline_bootstrap as bootstrap


def write_manifest(root: Path, assets: list[dict], *, total: int | None = None) -> Path:
    manifest = root / "device" / "models.manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "manifest_version": 1,
        "asset_count": len(assets),
        "total_bytes": sum(item.get("bytes", 0) for item in assets) if total is None else total,
        "assets": assets,
    }), encoding="utf-8")
    return manifest


def asset(path: str, content: bytes, *, required: bool = True) -> dict:
    return {
        "stage": "test",
        "path": path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "required": required,
    }


def test_full_manifest_verification(tmp_path: Path) -> None:
    content = b"voiceai-model"
    model = tmp_path / "data/raw/models/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(content)
    manifest = write_manifest(tmp_path, [asset("data/raw/models/model.bin", content)])

    report = bootstrap.verify_models(manifest, tmp_path, profile="manifest")

    assert report.ok
    assert report.checked_assets == 1
    assert report.checked_bytes == len(content)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    content = b"escape"
    manifest = write_manifest(tmp_path, [asset("../escape.bin", content)])

    report = bootstrap.verify_models(manifest, tmp_path, profile="manifest")

    assert not report.ok
    assert any(item.code == "manifest_path" for item in report.findings)


def test_six_direction_requires_optional_assets(tmp_path: Path) -> None:
    content = b"optional"
    manifest = write_manifest(
        tmp_path,
        [asset("data/raw/models/optional.bin", content, required=False)],
    )

    strict = bootstrap.verify_models(manifest, tmp_path, profile="six-direction")
    core = bootstrap.verify_models(manifest, tmp_path, profile="core")

    assert not strict.ok
    assert core.ok
    assert any(item.code == "profile_component_missing" for item in strict.findings)
    assert any(item.level == "warning" for item in core.findings)


def test_hash_mismatch_fails_even_when_size_matches(tmp_path: Path) -> None:
    expected = b"abc"
    model = tmp_path / "data/raw/models/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"xyz")
    manifest = write_manifest(tmp_path, [asset("data/raw/models/model.bin", expected)])

    full = bootstrap.verify_models(manifest, tmp_path, profile="manifest")
    quick = bootstrap.verify_models(manifest, tmp_path, quick=True, profile="manifest")

    assert not full.ok
    assert any(item.code == "model_hash" for item in full.findings)
    assert quick.ok


def test_incomplete_download_fails(tmp_path: Path) -> None:
    content = b"valid"
    model = tmp_path / "data/raw/models/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(content)
    (model.parent / "weights.bin.incomplete").write_bytes(b"")
    manifest = write_manifest(tmp_path, [asset("data/raw/models/model.bin", content)])

    report = bootstrap.verify_models(manifest, tmp_path, quick=True, profile="manifest")

    assert not report.ok
    assert any(item.code == "incomplete_download" for item in report.findings)


def test_policy_is_fail_closed() -> None:
    unsafe = bootstrap.verify_offline_policy({})
    safe = bootstrap.verify_offline_policy(dict(bootstrap.SAFE_RUNTIME_ENV))

    assert not unsafe.ok
    assert safe.ok
    assert len(unsafe.findings) == len(bootstrap.SAFE_RUNTIME_ENV)


def test_requirements_reject_unpinned_lines(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2\ntorch==2.11.0\n", encoding="utf-8")

    pins, findings = bootstrap.parse_requirements(requirements)

    assert pins == {"torch": "2.11.0"}
    assert any(item.code == "requirement_unpinned" for item in findings)


def test_install_command_cannot_use_network(tmp_path: Path) -> None:
    command = bootstrap.pip_install_command(
        sys.executable,
        tmp_path / "wheelhouse",
        tmp_path / "requirements.txt",
        dry_run=True,
    )

    assert "--no-index" in command
    assert "--find-links" in command
    assert "--dry-run" in command
    assert not any(token.startswith("http://") or token.startswith("https://") for token in command)


def test_existing_non_venv_path_is_refused(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==2.2.6\n", encoding="utf-8")
    destination = tmp_path / "environment"
    destination.mkdir()

    with pytest.raises(bootstrap.PreflightError, match="non-venv"):
        bootstrap.validate_install_inputs(wheelhouse, requirements, destination)
