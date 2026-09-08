"""Explicit, resumable installation of AI model files from a manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_RANGE_BYTES = 4 * 1024 * 1024
DEFAULT_RETRIES = 3
DEFAULT_WORKERS = 8


class ModelInstallError(Exception):
    """Raised when a model cannot be downloaded or does not verify."""


#: `sha256` 取该字面量表示「延迟校验」：用户信任来源，安装时跳过 size / sha256
#: 校验，仅按 source 下载并落盘（大模型哈希易超时，或权重地址待定）。与
#: `audio_ai_hifigan` worker 的 MODEL_HASH_DEFERRED 同源同义。用非 hex 字符串，
#: 避免与全零占位哈希混淆（后者仍按普通哈希参与校验）。
MODEL_HASH_DEFERRED = "deferred"


def read_model_entry(model_dir: Path, model_id: str) -> dict[str, Any]:
    """读取清单条目，返回其中全部权重文件（`artifacts`）。

    兼容两种形态：

    - 单文件模型：使用顶层 `file` / `source` / `sha256` / `size_bytes`
      （DeepFilterNet、AudioSR 即如此）。
    - 多文件模型：条目含 `files` 数组（FlashSR 需要三个权重），此时以
      `files` 为准，忽略顶层同名字段。

    返回的 `artifacts` 中每项含 `name` / `file` / `size_bytes` /
    `sha256` / `source`。
    """
    manifest_path = model_dir / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload["models"]
        entry = next(item for item in entries if item["id"] == model_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration) as error:
        raise ModelInstallError(f"model is not in manifest: {model_id}") from error
    if not isinstance(entry, dict):
        raise ModelInstallError(f"invalid manifest entry: {model_id}")
    return {**entry, "model_id": model_id, "artifacts": _parse_artifacts(entry, model_id)}


def _parse_artifacts(entry: dict[str, Any], model_id: str) -> list[dict[str, Any]]:
    files = entry.get("files")
    if files is None:
        return [_validate_artifact(dict(entry), model_id)]
    if not isinstance(files, list) or not files:
        raise ModelInstallError(f"invalid manifest files list: {model_id}")
    return [_validate_artifact(dict(item), model_id) for item in files]


def _validate_artifact(raw: dict[str, Any], model_id: str) -> dict[str, Any]:
    for field in ("file", "source", "sha256", "size_bytes"):
        if field not in raw:
            raise ModelInstallError(f"manifest field is missing: {field}")
    try:
        size_bytes = int(raw["size_bytes"])
    except (TypeError, ValueError) as error:
        raise ModelInstallError(f"invalid model size: {model_id}") from error
    source = str(raw["source"])
    if not source.startswith("https://"):
        raise ModelInstallError("model source must use HTTPS")
    digest = str(raw["sha256"]).lower()
    if digest == MODEL_HASH_DEFERRED:
        # 用户信任来源：安装时仅下载落盘，跳过 size 与 sha256 校验。
        if size_bytes < 0:
            raise ModelInstallError(f"invalid model size: {model_id}")
        return {
            "name": str(raw.get("name", "")),
            "file": str(raw["file"]),
            "size_bytes": size_bytes,
            "sha256": digest,
            "source": source,
        }
    if size_bytes <= 0:
        raise ModelInstallError(f"invalid model size: {model_id}")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ModelInstallError(f"invalid model SHA-256: {model_id}")
    return {
        "name": str(raw.get("name", "")),
        "file": str(raw["file"]),
        "size_bytes": size_bytes,
        "sha256": digest,
        "source": source,
    }


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


class ProgressReporter:
    def __init__(self, total: int, initial: int = 0) -> None:
        self.total = total
        self.downloaded = initial
        self.last_reported = initial
        self.lock = threading.Lock()

    def add(self, amount: int) -> None:
        with self.lock:
            self.downloaded += amount
            if (
                self.downloaded - self.last_reported < 16 * 1024 * 1024
                and self.downloaded < self.total
            ):
                return
            self.last_reported = self.downloaded
            report_progress(self.downloaded, self.total)


def _download_range(
    source: str,
    part_path: Path,
    start: int,
    end: int,
    expected_size: int,
    retries: int,
    chunk_bytes: int,
    progress: ProgressReporter,
) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    range_size = end - start
    for attempt in range(retries):
        existing_size = part_path.stat().st_size if part_path.exists() else 0
        if existing_size > range_size:
            part_path.unlink()
            existing_size = 0
        if existing_size == range_size:
            return
        request = urllib.request.Request(source)
        request.add_header("Range", f"bytes={start + existing_size}-{end - 1}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status != 206:
                    raise ModelInstallError(
                        f"server did not honor byte range for {part_path.name}: HTTP {status}"
                    )
                mode = "ab" if existing_size else "wb"
                downloaded = existing_size
                with part_path.open(mode) as destination:
                    while True:
                        remaining = range_size - downloaded
                        if remaining <= 0:
                            break
                        block = response.read(min(chunk_bytes, remaining))
                        if not block:
                            break
                        destination.write(block)
                        downloaded += len(block)
                        progress.add(len(block))
                    destination.flush()
                    os.fsync(destination.fileno())
                if downloaded != range_size:
                    raise ModelInstallError(
                        f"incomplete range: {downloaded}/{range_size} bytes"
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


def _download_parts(
    source: str,
    part_path: Path,
    parts_dir: Path,
    expected_size: int,
    retries: int,
    chunk_bytes: int,
    workers: int,
) -> list[Path]:
    ranges: list[tuple[int, int, Path]] = []
    start = 0
    index = 0
    while start < expected_size:
        end = min(expected_size, start + DEFAULT_RANGE_BYTES)
        range_path = part_path if index == 0 else parts_dir / f"{index:08d}.part"
        ranges.append((start, end, range_path))
        start = end
        index += 1

    parts_dir.mkdir(parents=True, exist_ok=True)
    for _, end, range_path in ranges:
        range_size = end - next(item[0] for item in ranges if item[2] == range_path)
        if range_path.exists() and range_path.stat().st_size > range_size:
            range_path.unlink()

    initial = sum(path.stat().st_size for _, _, path in ranges if path.exists())
    progress = ProgressReporter(expected_size, initial)
    report_progress(initial, expected_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(
                _download_range,
                source,
                range_path,
                start,
                end,
                expected_size,
                retries,
                chunk_bytes,
                progress,
            )
            for start, end, range_path in ranges
        ]
        for future in futures:
            future.result()
    return [path for _, _, path in ranges]


def report_progress(downloaded: int, total: int) -> None:
    percent = min(100.0, downloaded * 100.0 / total) if total else 0.0
    print(f"progress {percent:.2f}% ({downloaded}/{total} bytes)", file=sys.stderr, flush=True)


def install_model(
    model_dir: Path,
    model_id: str,
    retries: int = DEFAULT_RETRIES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    workers: int = DEFAULT_WORKERS,
) -> list[Path]:
    """下载并校验一个模型条目。

    支持单文件与多文件（`files[]`，见 `read_model_entry`）。返回已安装文件
    的路径列表。**已存在的文件直接跳过**（按 size+sha256 判定），因此支持
    断点续传：重跑安装器时只补回缺失或损坏的文件。

    进度以 `model n/N: name` + `progress % (bytes/total)` 行输出到 stderr，
    供 Rust 侧 `install_model` 透传给前端。
    """
    entry = read_model_entry(model_dir, model_id)
    artifacts = entry["artifacts"]
    total_files = len(artifacts)
    installed: list[Path] = []
    for index, artifact in enumerate(artifacts, start=1):
        print(f"model {index}/{total_files}: {artifact['name'] or artifact['file']}",
              file=sys.stderr, flush=True)
        installed.append(
            _install_artifact(model_dir, artifact, retries, chunk_bytes, workers)
        )
    return installed


def _install_artifact(
    model_dir: Path,
    artifact: dict[str, Any],
    retries: int,
    chunk_bytes: int,
    workers: int,
) -> Path:
    target = _resolve_target(model_dir, artifact)
    expected_size = artifact["size_bytes"]
    expected_sha256 = artifact["sha256"]
    if target.is_file() and (
        expected_sha256 == MODEL_HASH_DEFERRED or verify_file(target, expected_size, expected_sha256)
    ):
        print(f"model already installed: {target}", file=sys.stderr, flush=True)
        return target

    part_path = target.with_name(target.name + ".part")
    parts_dir = target.with_name(
        target.name + f".parts.{expected_sha256[:16]}.{DEFAULT_RANGE_BYTES}"
    )
    part_paths = _download_parts(
        str(artifact["source"]),
        part_path,
        parts_dir,
        expected_size,
        max(1, retries),
        max(1, chunk_bytes),
        max(1, workers),
    )
    merged_path = target.with_name(target.name + ".merge.part")
    try:
        with merged_path.open("wb") as merged:
            for index, path in enumerate(part_paths):
                expected_part_size = min(
                    DEFAULT_RANGE_BYTES,
                    expected_size - index * DEFAULT_RANGE_BYTES,
                )
                if not path.is_file() or path.stat().st_size != expected_part_size:
                    raise ModelInstallError(
                        f"downloaded model part is incomplete: {path.name}"
                    )
                with path.open("rb") as source:
                    shutil.copyfileobj(source, merged, length=chunk_bytes)
            merged.flush()
            os.fsync(merged.fileno())
    except Exception:
        if merged_path.exists():
            merged_path.unlink()
        raise
    if expected_sha256 != MODEL_HASH_DEFERRED:
        if merged_path.stat().st_size != expected_size:
            raise ModelInstallError("downloaded model has an unexpected size")
        actual_sha256 = sha256_file(merged_path)
        if actual_sha256 != expected_sha256:
            merged_path.unlink()
            raise ModelInstallError(
                f"model hash mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
    elif expected_size and merged_path.stat().st_size != expected_size:
        raise ModelInstallError("downloaded model has an unexpected size")
    os.replace(merged_path, target)
    for path in part_paths:
        if path.exists():
            path.unlink()
    if parts_dir.exists():
        parts_dir.rmdir()
    print(f"model installed: {target}", file=sys.stderr, flush=True)
    return target


def _resolve_target(model_dir: Path, artifact: dict[str, Any]) -> Path:
    target = (model_dir / str(artifact["file"])).resolve()
    model_root = model_dir.resolve()
    if model_root not in target.parents:
        raise ModelInstallError("model file escapes model directory")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Install an AI model from models/manifest.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    model_dir = Path(
        args.model_dir
        or os.environ.get("AUDIO_AI_MODEL_DIR")
        or Path(__file__).resolve().parents[2] / "models"
    )
    try:
        installed = install_model(
            model_dir,
            args.model_id,
            retries=args.retries,
            workers=args.workers,
        )
    except ModelInstallError as error:
        print(f"model installation failed: {error}", file=sys.stderr)
        return 1
    print(f"model installation complete: {len(installed)} file(s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
