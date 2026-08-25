"""Explicit, resumable installation of AI model files from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_RETRIES = 3


class ModelInstallError(Exception):
    """Raised when a model cannot be downloaded or does not verify."""


def read_model_entry(model_dir: Path, model_id: str) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload["models"]
        entry = next(item for item in entries if item["id"] == model_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as error:
        raise ModelInstallError(f"model is not in manifest: {model_id}") from error
    if not isinstance(entry, dict):
        raise ModelInstallError(f"invalid manifest entry: {model_id}")
    for field in ("file", "source", "sha256", "size_bytes"):
        if field not in entry:
            raise ModelInstallError(f"manifest field is missing: {field}")
    try:
        size_bytes = int(entry["size_bytes"])
    except (TypeError, ValueError) as error:
        raise ModelInstallError(f"invalid model size: {model_id}") from error
    if size_bytes <= 0:
        raise ModelInstallError(f"invalid model size: {model_id}")
    source = str(entry["source"])
    if not source.startswith("https://"):
        raise ModelInstallError("model source must use HTTPS")
    digest = str(entry["sha256"]).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ModelInstallError(f"invalid model SHA-256: {model_id}")
    return {**entry, "size_bytes": size_bytes, "sha256": digest, "source": source}


def sha256_file(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def _download_part(
    source: str,
    part_path: Path,
    expected_size: int,
    retries: int,
    chunk_bytes: int,
) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        if existing_size > expected_size:
            part_path.unlink()
            existing_size = 0
        request = urllib.request.Request(source)
        if existing_size:
            request.add_header("Range", f"bytes={existing_size}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                resumed = existing_size > 0 and status == 206
                if existing_size and not resumed:
                    existing_size = 0
                mode = "ab" if resumed else "wb"
                downloaded = existing_size
                with part_path.open(mode) as destination:
                    while True:
                        block = response.read(chunk_bytes)
                        if not block:
                            break
                        destination.write(block)
                        downloaded += len(block)
                        report_progress(downloaded, expected_size)
                    destination.flush()
                    os.fsync(destination.fileno())
                if downloaded != expected_size:
                    raise ModelInstallError(
                        f"incomplete download: {downloaded}/{expected_size} bytes"
                    )
                return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ModelInstallError) as error:
            if attempt + 1 == retries:
                raise ModelInstallError(str(error)) from error
            print(
                f"download interrupted, retrying ({attempt + 1}/{retries}): {error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(1.0 + attempt)
    raise ModelInstallError("download failed")


def report_progress(downloaded: int, total: int) -> None:
    percent = min(100.0, downloaded * 100.0 / total) if total else 0.0
    print(f"progress {percent:.2f}% ({downloaded}/{total} bytes)", file=sys.stderr, flush=True)


def install_model(
    model_dir: Path,
    model_id: str,
    retries: int = DEFAULT_RETRIES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Path:
    entry = read_model_entry(model_dir, model_id)
    target = (model_dir / str(entry["file"])).resolve()
    model_root = model_dir.resolve()
    if model_root not in target.parents:
        raise ModelInstallError("model file escapes model directory")
    expected_size = int(entry["size_bytes"])
    expected_sha256 = str(entry["sha256"])
    if verify_file(target, expected_size, expected_sha256):
        print(f"model already installed: {target}", file=sys.stderr, flush=True)
        return target

    part_path = target.with_name(target.name + ".part")
    _download_part(
        str(entry["source"]),
        part_path,
        expected_size,
        max(1, retries),
        max(1, chunk_bytes),
    )
    if part_path.stat().st_size != expected_size:
        raise ModelInstallError("downloaded model has an unexpected size")
    actual_sha256 = sha256_file(part_path)
    if actual_sha256 != expected_sha256:
        raise ModelInstallError(
            f"model hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(part_path, target)
    print(f"model installed: {target}", file=sys.stderr, flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Install an AI model from models/manifest.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args()
    model_dir = Path(args.model_dir or Path(__file__).resolve().parents[2] / "models")
    try:
        install_model(model_dir, args.model_id, retries=args.retries)
    except ModelInstallError as error:
        print(f"model installation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
