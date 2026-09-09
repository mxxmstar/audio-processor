"""JSONL 协议原语：事件输出、错误类型与外部二进制定位。

本模块被 `worker.py`（协议主循环）与 `pipeline.py`（重量级实现）共用。
两者都必须经由 `sys.modules["protocol"]` 拿到**同一个** `WorkerFailure`
类对象，否则 `worker.py` 里的 `except WorkerFailure` 匹配不上
`pipeline.py` 抛出的异常 —— Rust 以 `python <绝对路径>/worker.py` 启动本
模块时没有包上下文，若把协议原语留在 `worker.py` 里，`import worker` 会
产生与 `__main__` 不同的第二个模块对象。

本模块只依赖标准库，可以安全地在 ready 握手之前导入。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
SCRIPT_ROOT = Path(__file__).resolve().parents[2]

EMIT_LOCK = threading.Lock()


class WorkerFailure(Exception):
    """可被协议识别的错误：带有供 Rust 侧分类的 `code`。"""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _force_utf8_stdio() -> None:
    """强制标准流使用 UTF-8。

    Rust 端以 UTF-8 字节写入 JSONL 请求，但 Windows 上 sys.stdin 默认按区域
    编码（如 cp936/GBK）解码。当图片路径含非 ASCII 字符（日文、罕见汉字等）时
    解码会失败或产生乱码，导致 Worker 判定「输入图片不存在」。
    stdout 同理，因为进度消息使用 ensure_ascii=False 输出中文。
    """
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def log(message: str) -> None:
    """把关键诊断信息写入 stderr。

    stdout 独占于 JSONL 协议，不能混入任何非协议输出；Rust 侧会完整采集
    stderr，因此 stderr 是定位「进程异常退出但没有 traceback」的唯一通道。
    """
    print(message, file=sys.stderr, flush=True)


def read_stdin_line() -> str | None:
    """安全读取一行协议输入。

    直接 `for line in sys.stdin` 迭代时，解码错误或 I/O 错误会以异常形式冲出
    主循环：在途任务被杀死、进程以退出码 1 结束，且协议上收不到任何事件。
    这里把这类错误记录到 stderr 并视为输入结束，让主循环正常收尾。
    """
    try:
        return sys.stdin.readline()
    except (UnicodeDecodeError, OSError) as error:
        log(f"stdin read failed: {type(error).__name__}: {error}\n{traceback.format_exc()}")
        return None


def emit(payload: dict[str, Any]) -> None:
    with EMIT_LOCK:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def emit_progress(
    request_id: str,
    phase: str,
    percent: float,
    processed_tiles: int | None = None,
    total_tiles: int | None = None,
    message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "progress",
        "phase": phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    if processed_tiles is not None:
        payload["processed_tiles"] = int(processed_tiles)
    if total_tiles is not None:
        payload["total_tiles"] = int(total_tiles)
    if message is not None:
        payload["message"] = message
    emit(payload)


def emit_error(request_id: str, failure: WorkerFailure) -> None:
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": "error",
            "code": failure.code,
            "message": str(failure),
            "retryable": failure.retryable,
        }
    )


def resolve_binary(name: str, env_name: str) -> str:
    """定位外部二进制（环境变量优先，其次仓库 bin/，最后 PATH）。

    图像域默认不需要 ffmpeg；保留此工具仅为与音频侧协议工具保持对称。
    """
    configured = os.environ.get(env_name)
    if configured:
        return configured
    bundled = SCRIPT_ROOT / "bin" / (f"{name}.exe" if os.name == "nt" else name)
    if bundled.is_file():
        return str(bundled)
    found = shutil.which(name)
    if found:
        return found
    raise WorkerFailure("RUNTIME_NOT_FOUND", f"cannot find {name}")
