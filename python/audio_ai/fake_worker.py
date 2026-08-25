"""Deterministic AI worker used to validate the Rust subprocess protocol.

This worker deliberately does not load an audio model. It is only a local
protocol fixture for the AI-1 implementation and is not shipped as the
production enhancement backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any


PROTOCOL_VERSION = 1


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def emit_error(request_id: str, code: str, message: str) -> None:
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": "error",
            "code": code,
            "message": message,
            "retryable": False,
        }
    )


def process_request(
    request: dict[str, Any], mode: str, cancel_event: threading.Event
) -> None:
    request_id = str(request.get("request_id", ""))
    if mode == "error":
        emit_error(request_id, "INFERENCE_FAILED", "fake worker inference failure")
        return
    if mode == "invalid":
        sys.stdout.write("this is not json\n")
        sys.stdout.flush()
        return

    steps = 200 if mode == "slow" else 4
    delay = 0.02 if mode == "slow" else 0.01
    for index in range(steps + 1):
        if cancel_event.is_set():
            emit_error(request_id, "CANCELLED", "fake worker cancelled")
            return
        percent = index * 100.0 / steps
        emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "type": "progress",
                "phase": "inference",
                "percent": percent,
                "processed_seconds": float(index),
                "total_seconds": float(steps),
            }
        )
        if index != steps:
            time.sleep(delay)

    output_path = str(request.get("output_path", ""))
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as output:
            output.write(b"fake-ai-output\n")

    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": "result",
            "status": "completed",
            "output_path": output_path,
            "model_id": "fake-model",
            "model_version": "0.1.0",
            "sample_rate": 48000,
            "channels": 2,
            "duration_seconds": float(steps),
            "peak_db": -1.0,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("success", "slow", "error", "invalid"), default="success"
    )
    args = parser.parse_args()

    request_id = ""
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": "ready",
            "worker_version": "fake-0.1.0",
            "models": ["fake-model"],
        }
    )

    active_thread: threading.Thread | None = None
    cancel_event = threading.Event()
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit_error(request_id, "INVALID_REQUEST", "request is not valid JSON")
            continue

        request_id = str(request.get("request_id", request_id))
        command = request.get("command")
        if command == "process":
            if active_thread and active_thread.is_alive():
                emit_error(request_id, "BUSY", "fake worker is already processing")
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
            emit_error(request_id, "INVALID_COMMAND", f"unsupported command: {command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
