"""协议自检用的假后端，不加载任何模型。

用途：在没有 3.3 GB 权重、也没有 GPU 的机器上验证 JSONL 协议的四种事件
（ready / progress / result / error）与取消路径，以及后续 Rust 侧集成。
与 `python/audio_ai/fake_worker.py` 的定位相同，仅供开发期使用。

    python python/audio_ai_flashsr/fake_worker.py --mode success
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import PROTOCOL_VERSION, WorkerFailure, emit, emit_error  # noqa: E402

WORKER_VERSION = "fake-flashsr-0.1.0"

MODES = ("success", "slow", "error", "crash")


def process_request(request: dict[str, Any], mode: str, cancel_event: threading.Event) -> None:
    request_id = str(request.get("request_id", ""))
    if mode == "crash":
        print("fake flashsr worker process crash", file=sys.stderr, flush=True)
        os._exit(1)
    if mode == "error":
        emit_error(request_id, WorkerFailure("INFERENCE_FAILED", "fake flashsr inference failure"))
        return

    steps = 40 if mode == "slow" else 6
    delay = 0.05 if mode == "slow" else 0.01
    for index in range(steps + 1):
        if cancel_event.is_set():
            emit_error(request_id, WorkerFailure("CANCELLED", "fake flashsr cancelled"))
            return
        emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "type": "progress",
                "phase": "inference",
                "percent": index * 100.0 / steps,
                "processed_seconds": float(index) * 0.128,
                "total_seconds": float(steps) * 0.128,
            }
        )
        if index != steps:
            time.sleep(delay)

    output_path = str(request.get("output_path", ""))
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-flashsr-output\n")

    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": "result",
            "status": "completed",
            "output_path": output_path,
            "model_id": "flashsr",
            "model_version": "fake",
            "sample_rate": 48000,
            "channels": 2,
            "duration_seconds": float(steps) * 0.128,
            "peak_db": -1.0,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashSR protocol fixture worker")
    parser.add_argument("--mode", choices=MODES, default="success")
    args = parser.parse_args()

    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "",
            "type": "ready",
            "worker_version": WORKER_VERSION,
            "models": ["flashsr"],
            "model_errors": [],
        }
    )

    active_thread: threading.Thread | None = None
    cancel_event = threading.Event()
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit_error("", WorkerFailure("INVALID_REQUEST", "request is not valid JSON"))
            continue

        request_id = str(request.get("request_id", ""))
        command = request.get("command")
        if command == "process":
            if active_thread and active_thread.is_alive():
                emit_error(
                    request_id, WorkerFailure("BUSY", "fake flashsr worker is already processing")
                )
                continue
            cancel_event.clear()
            active_thread = threading.Thread(
                target=process_request, args=(request, args.mode, cancel_event), daemon=True
            )
            active_thread.start()
        elif command == "cancel":
            cancel_event.set()
        elif command == "shutdown":
            return 0
        else:
            emit_error(
                request_id, WorkerFailure("INVALID_COMMAND", f"unsupported command: {command}")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
