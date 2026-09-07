"""FlashSR 音频超分 Worker：JSONL 协议入口与进程生命周期。

本模块顶层**只导入标准库**。Rust 侧的 ready 握手超时为 3 秒
（`src-tauri/src/audio_quality/ai_worker.rs` 的 `READY_TIMEOUT`），而
`numpy` + `torch` + `FlashSR` 的导入实测约 6.6 秒，因此重量级依赖必须延迟到
收到 `process` 命令之后再加载。详见实施计划风险 R12。

所有 numpy / torch 相关实现放在同目录的 `pipeline.py`，由
`_load_pipeline()` 在任务线程启动前导入；协议原语在 `protocol.py`。

协议与 `python/audio_ai/worker.py` 完全一致（v1）：stdin 收 JSONL 命令
（`process` / `cancel` / `shutdown`），stdout 发 JSONL 事件
（`ready` / `progress` / `result` / `error`）。stdout 被协议独占，
任何诊断输出都必须写 stderr。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, NamedTuple

WORKER_VERSION = "python-flashsr-0.1.0"
MODULE_ROOT = Path(__file__).resolve().parent

# Rust 以 `python <绝对路径>/worker.py` 启动本模块，此时没有包上下文，相对
# 导入不可用。把本目录加入 sys.path，让 protocol / pipeline / backend 都能
# 以顶层模块解析。这一步必须早于下面的 import，且必须发生在模块顶层 ——
# 这样无论以脚本、`python -m audio_ai_flashsr` 还是被测试导入，路径都就绪。
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    SCRIPT_ROOT,
    WorkerFailure,
    emit,
    emit_error,
    log,
    read_stdin_line,
)

#: 超过此体积的模型在启动时不算 SHA-256，否则会超出 Rust 侧的 ready 超时。
#: 该校验在 find_model 内、加载模型前必定执行。
STARTUP_HASH_MAX_BYTES = 128 * 1024 * 1024

SUPPORTED_BACKENDS = frozenset({"flashsr"})


# --------------------------------------------------------------------------
# 模型清单
# --------------------------------------------------------------------------


class Artifact(NamedTuple):
    """清单中的一个权重文件。"""

    name: str
    file: str
    size_bytes: int
    sha256: str
    source: str


class ModelSpec(NamedTuple):
    """解析并校验后的模型描述。

    `artifacts` 按清单中的声明顺序保存 `(名称, 绝对路径)`。FlashSR 需要三个
    权重，取值时按名称显式挑选（见 `ordered_weights`），不依赖清单顺序。
    """

    model_id: str
    version: str
    backend: str
    model_name: str
    artifacts: tuple[tuple[str, Path], ...]


def ordered_weights(spec: ModelSpec, expected: tuple[str, ...]) -> tuple[Path, ...]:
    """按给定名称顺序取出权重路径，缺项时抛 WorkerFailure。"""
    lookup = dict(spec.artifacts)
    missing = [name for name in expected if name not in lookup]
    if missing:
        raise WorkerFailure(
            "MODEL_MANIFEST_INVALID",
            f"model {spec.model_id} is missing weight entries: {', '.join(missing)}",
        )
    return tuple(lookup[name] for name in expected)


def _parse_artifact(entry: dict[str, Any]) -> Artifact:
    file_value = entry.get("file")
    if not file_value:
        raise ValueError("model file is required")
    raw_size = entry.get("size_bytes")
    size_bytes = 0
    if raw_size not in (None, ""):
        try:
            size_bytes = int(raw_size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid size_bytes: {raw_size}") from error
        if size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")
    sha256 = str(entry.get("sha256", "")).lower()
    if not sha256:
        raise ValueError("model sha256 is required")
    return Artifact(
        name=str(entry.get("name", "") or ""),
        file=str(file_value),
        size_bytes=size_bytes,
        sha256=sha256,
        source=str(entry.get("source", "")),
    )


def _parse_artifacts(entry: dict[str, Any]) -> tuple[Artifact, ...]:
    """解析 `files[]`，无该字段时按单文件模型处理。

    为了与 `audio_ai` 的清单格式兼容：不带 `files` 的条目行为完全不变；
    带 `files` 时以 `files` 为准，忽略顶层的 size_bytes / sha256 / source。
    """
    files = entry.get("files")
    if files is None:
        return (_parse_artifact(entry),)
    if not isinstance(files, list) or not files:
        raise ValueError("model files must be a non-empty list")
    return tuple(_parse_artifact(item) for item in files)


def read_manifest(model_dir: Path) -> dict[str, dict[str, Any]] | None:
    manifest_path = Path(
        os.environ.get("AUDIO_AI_MODEL_MANIFEST", str(model_dir / "manifest.json"))
    )
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("models", [])
        if not isinstance(entries, list):
            raise ValueError("models must be a list")
        result: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("model entry must be an object")
            model_id = str(entry["id"])
            backend = str(entry.get("backend", "torchscript"))
            # 共享 manifest 含其它后端的条目（如 audiosr / deepfilternet）；
            # 本 Worker 只收录自己支持的后端，其余跳过，避免整份清单被拒。
            if backend not in SUPPORTED_BACKENDS:
                continue
            result[model_id] = {
                "file": str(entry["file"]),
                "version": str(entry.get("version", "unknown")),
                "backend": backend,
                "model_name": str(entry.get("model_name", "")),
                "artifacts": _parse_artifacts(entry),
            }
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise WorkerFailure("MODEL_MANIFEST_INVALID", str(error)) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(model_dir: Path, artifact: Artifact) -> Path:
    candidate = (model_dir / artifact.file).resolve()
    if model_dir.resolve() not in candidate.parents:
        raise WorkerFailure("MODEL_MANIFEST_INVALID", "model file escapes model directory")
    return candidate


def find_model(model_id: str, model_dir: Path) -> ModelSpec:
    manifest = read_manifest(model_dir)
    if manifest is None:
        raise WorkerFailure(
            "MODEL_NOT_FOUND", f"model manifest is missing: {model_dir / 'manifest.json'}"
        )
    entry = manifest.get(model_id)
    if entry is None:
        raise WorkerFailure("MODEL_NOT_FOUND", f"model is not in manifest: {model_id}")

    artifacts: list[tuple[str, Path]] = []
    for artifact in entry["artifacts"]:
        candidate = resolve_artifact(model_dir, artifact)
        if not candidate.is_file():
            raise WorkerFailure(
                "MODEL_NOT_FOUND", f"model file does not exist: {candidate.name}"
            )
        if artifact.size_bytes and candidate.stat().st_size != artifact.size_bytes:
            raise WorkerFailure(
                "MODEL_SIZE_MISMATCH", f"model size mismatch: {candidate.name}"
            )
        if sha256_file(candidate) != artifact.sha256:
            raise WorkerFailure(
                "MODEL_HASH_MISMATCH", f"model hash mismatch: {candidate.name}"
            )
        artifacts.append((artifact.name, candidate))
    return ModelSpec(
        model_id=model_id,
        version=entry["version"],
        backend=entry["backend"],
        model_name=entry["model_name"],
        artifacts=tuple(artifacts),
    )


def available_models(model_dir: Path) -> tuple[list[str], list[str]]:
    try:
        manifest = read_manifest(model_dir)
    except WorkerFailure as failure:
        return [], [f"{failure.code}: {failure}"]
    if manifest is None:
        return [], []

    models: list[str] = []
    errors: list[str] = []
    for model_id, entry in manifest.items():
        problem: str | None = None
        deferred = False
        try:
            for artifact in entry["artifacts"]:
                candidate = resolve_artifact(model_dir, artifact)
                if not candidate.is_file():
                    problem = f"{model_id}: MODEL_NOT_FOUND"
                    break
                if artifact.size_bytes and candidate.stat().st_size != artifact.size_bytes:
                    problem = f"{model_id}: MODEL_SIZE_MISMATCH"
                    break
                if candidate.stat().st_size > STARTUP_HASH_MAX_BYTES:
                    # 多 GB 的 SHA-256 会超出 Rust 侧的 ready 超时；同样的
                    # 校验在 find_model 内、模型加载前必定执行。
                    deferred = True
                    continue
                if sha256_file(candidate) != artifact.sha256:
                    problem = f"{model_id}: MODEL_HASH_MISMATCH"
                    break
        except WorkerFailure as failure:
            # 单个条目异常（如权重路径逃逸）不能让整个 Worker 起不来：
            # 那会导致 ready 事件发不出去，Rust 侧表现为握手超时。
            problem = f"{model_id}: {failure.code}"
        if problem is not None:
            errors.append(problem)
            continue
        if deferred:
            errors.append(f"{model_id}: MODEL_HASH_DEFERRED")
        models.append(model_id)
    return sorted(models), sorted(errors)


# --------------------------------------------------------------------------
# 进程生命周期
# --------------------------------------------------------------------------


def _load_pipeline() -> Any:
    """延迟导入重量级实现（R12）。

    这里会拉起 numpy（约 0.9 s）、torch（约 5.7 s）与 FlashSR 上游模块，
    合计约 6.6 s，远超 Rust 侧 3 秒的 ready 超时，因此只能在收到 `process`
    命令后调用 —— 此时 ready 事件早已发出，握手不受影响。
    """
    import pipeline  # type: ignore[import-not-found]

    return pipeline


def process_in_thread(
    pipeline: Any, request: dict[str, Any], cancel_event: threading.Event
) -> None:
    request_id = str(request.get("request_id", ""))
    try:
        result = pipeline.enhance(request, cancel_event)
        emit(result)
        log("result emitted")
    except WorkerFailure as failure:
        emit_error(request_id, failure)
    except BaseException as error:  # noqa: BLE001
        # 必须连 SystemExit / KeyboardInterrupt 一起兜住：它们是
        # BaseException 而非 Exception，漏掉会让线程静默死亡，进程随后以
        # 退出码 1 结束且协议上收不到任何 error，表现为「进程异常退出」。
        log(f"unhandled worker error:\n{traceback.format_exc()}")
        emit_error(
            request_id,
            WorkerFailure("INFERENCE_FAILED", f"{type(error).__name__}: {error}"),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashSR audio super-resolution worker")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    if args.model_dir:
        os.environ["AUDIO_AI_MODEL_DIR"] = args.model_dir
    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))

    # 清单异常绝不能阻止 ready 发出：Rust 侧只等 3 秒，收不到 ready 就会
    # 判定「AI 运行时不可用」，用户看不到任何具体原因。
    try:
        models, model_errors = available_models(model_dir)
    except WorkerFailure as failure:
        models, model_errors = [], [f"{failure.code}: {failure}"]
    except OSError as error:
        models, model_errors = [], [f"MODEL_MANIFEST_INVALID: {error}"]
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "",
            "type": "ready",
            "worker_version": WORKER_VERSION,
            "models": models,
            "model_errors": model_errors,
        }
    )

    active_thread: threading.Thread | None = None
    cancel_event = threading.Event()
    while True:
        line = read_stdin_line()
        if not line:  # EOF 或读取失败
            break
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit_error("", WorkerFailure("INVALID_REQUEST", "request is not valid JSON"))
            continue
        request_id = str(request.get("request_id", ""))
        command = request.get("command")
        if command == "process":
            if active_thread and active_thread.is_alive():
                emit_error(request_id, WorkerFailure("BUSY", "worker is already processing"))
                continue
            try:
                pipeline = _load_pipeline()
            except Exception as error:  # noqa: BLE001
                emit_error(
                    request_id,
                    WorkerFailure(
                        "MODEL_RUNTIME_NOT_FOUND",
                        f"cannot load FlashSR runtime: {type(error).__name__}: {error}",
                    ),
                )
                continue
            # 必须在主线程预热上游导入：它会经 joblib → loky 拉起
            # multiprocessing 的 resource tracker，若在推理线程内导入而主线程
            # 阻塞于 stdin 读取，会死锁（stdin 为管道时必现，见
            # pipeline.warm_up_imports 的说明）。预热失败不致命，交给线程内
            # 的常规错误处理路径上报。
            try:
                pipeline.warm_up_imports()
            except Exception as error:  # noqa: BLE001
                log(f"warm-up import failed: {type(error).__name__}: {error}")
            cancel_event.clear()
            # 必须用非守护线程：主线程一旦退出，守护线程会被立即杀死，
            # 正在写回的结果 / 错误事件会随之丢失，表现为「进程异常退出、
            # 退出码 1 且协议上收不到任何事件」，而文件其实已经生成成功。
            active_thread = threading.Thread(
                target=process_in_thread,
                args=(pipeline, request, cancel_event),
                daemon=False,
            )
            active_thread.start()
        elif command == "cancel":
            cancel_event.set()
        elif command == "shutdown":
            cancel_event.set()
            _join_active_thread(active_thread)
            return 0
        else:
            emit_error(
                request_id, WorkerFailure("INVALID_COMMAND", f"unsupported command: {command}")
            )
    # EOF：必须等待在途任务完成再返回。否则主线程先退出会让解释器开始关闭
    # （threading._shutdown 置 _SHUTTING_DOWN=True），仍在运行的推理线程内
    # 懒加载上游时，`joblib → loky → concurrent.futures.process` 会在导入期
    # 调用 threading._register_atexit() 并抛
    # "can't register atexit after shutdown"（阶段 6 实测）。
    _join_active_thread(active_thread)
    return 0


def _join_active_thread(active_thread: threading.Thread | None) -> None:
    """等待在途推理线程结束，避免解释器在其运行期间进入关闭流程。"""
    if active_thread is None:
        return
    if not active_thread.is_alive():
        return
    # 该线程是非守护线程，Python 原本也会在 _shutdown 中等待它；这里提前
    # join 只是为了让它在 _SHUTTING_DOWN 置位之前跑完。超时仅作兜底，避免
    # 极端情况下进程永久挂住。
    active_thread.join(timeout=3600.0)
    if active_thread.is_alive():
        log("timed out waiting for the in-flight task to finish")


if __name__ == "__main__":
    raise SystemExit(main())
