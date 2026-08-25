"""TorchScript-backed audio enhancement worker.

The model contract is intentionally small and explicit: a TorchScript module
receives a float32 tensor shaped [1, channels, samples] and returns a tensor
with the same shape. The model must be supplied separately; this worker does
not download weights and never treats resampling as AI enhancement.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_VERSION = 1
SCRIPT_ROOT = Path(__file__).resolve().parents[2]
EMIT_LOCK = threading.Lock()


class WorkerFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def emit(payload: dict[str, Any]) -> None:
    with EMIT_LOCK:
        sys.stdout.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()


def emit_progress(
    request_id: str,
    phase: str,
    percent: float,
    processed_seconds: float | None = None,
    total_seconds: float | None = None,
    message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "progress",
        "phase": phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    if processed_seconds is not None:
        payload["processed_seconds"] = float(processed_seconds)
    if total_seconds is not None:
        payload["total_seconds"] = float(total_seconds)
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


def probe_audio(path: str) -> tuple[int, int, float]:
    ffprobe = resolve_binary("ffprobe", "AUDIO_AI_FFPROBE")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkerFailure("UNSUPPORTED_INPUT", detail or "ffprobe failed")
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
        stream = payload["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = stream.get("duration") or payload.get("format", {}).get("duration")
        return sample_rate, channels, float(duration or 0.0)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise WorkerFailure("UNSUPPORTED_INPUT", f"invalid ffprobe output: {error}") from error


def decode_audio(path: str, sample_rate: int, channels: int) -> np.ndarray:
    ffmpeg = resolve_binary("ffmpeg", "AUDIO_AI_FFMPEG")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        path,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkerFailure("DECODE_FAILED", detail or "ffmpeg decode failed")
    values = np.frombuffer(result.stdout, dtype=np.float32)
    if values.size == 0 or values.size % channels != 0:
        raise WorkerFailure("DECODE_FAILED", "decoded PCM is empty or has invalid channel count")
    audio = values.reshape(-1, channels).T.copy()
    if not np.isfinite(audio).all():
        raise WorkerFailure("DECODE_FAILED", "decoded PCM contains NaN or Inf")
    return audio


def encode_audio(audio: np.ndarray, output_path: str, sample_rate: int, channels: int) -> None:
    suffix = output_path.lower()
    if suffix.endswith(".part"):
        suffix = suffix[:-5]
    if suffix.endswith(".flac"):
        codec = "flac"
    elif suffix.endswith(".wav"):
        codec = "pcm_s16le"
    else:
        raise WorkerFailure("UNSUPPORTED_OUTPUT", "AI master output must be WAV or FLAC")

    ffmpeg = resolve_binary("ffmpeg", "AUDIO_AI_FFMPEG")
    pcm = np.ascontiguousarray(audio.T, dtype=np.float32).tobytes()
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-c:a",
        codec,
        output_path,
    ]
    result = subprocess.run(command, input=pcm, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkerFailure("ENCODE_FAILED", detail or "ffmpeg encode failed")


def model_candidates(model_id: str, model_dir: Path) -> list[Path]:
    safe_name = Path(model_id).name
    return [model_dir / f"{safe_name}{suffix}" for suffix in (".ts", ".pt", ".pth")]


def find_model(model_id: str, model_dir: Path) -> Path:
    configured = os.environ.get("AUDIO_AI_MODEL")
    candidates = [Path(configured)] if configured else model_candidates(model_id, model_dir)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise WorkerFailure("MODEL_NOT_FOUND", f"no TorchScript model found for {model_id}")


def available_models(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        return []
    names = set()
    for path in model_dir.iterdir():
        if path.suffix.lower() in {".ts", ".pt", ".pth"}:
            names.add(path.stem)
    return sorted(names)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def infer_chunk(model: torch.jit.ScriptModule, chunk: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
    try:
        with torch.inference_mode():
            output = model(tensor)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WorkerFailure("OUT_OF_MEMORY", str(error)) from error
        raise WorkerFailure("INFERENCE_FAILED", str(error)) from error
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise WorkerFailure("INFERENCE_FAILED", "model output is not a tensor")
    if output.ndim == 2:
        output = output.unsqueeze(0)
    if output.ndim != 3 or output.shape[0] != 1:
        raise WorkerFailure("INFERENCE_FAILED", "model output must have shape [1, channels, samples]")
    result = output.detach().float().cpu().numpy()[0]
    if result.shape != chunk.shape:
        raise WorkerFailure(
            "INFERENCE_FAILED",
            f"model output shape {result.shape} does not match input {chunk.shape}",
        )
    if not np.isfinite(result).all():
        raise WorkerFailure("OUTPUT_INVALID", "model output contains NaN or Inf")
    return result


def enhance(request: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    request_id = str(request.get("request_id", ""))
    input_path = str(request.get("input_path", ""))
    output_path = str(request.get("output_path", ""))
    model_id = str(request.get("model_id", ""))
    if not input_path or not Path(input_path).is_file():
        raise WorkerFailure("UNSUPPORTED_INPUT", "input audio does not exist")
    if not output_path or Path(output_path).resolve() == Path(input_path).resolve():
        raise WorkerFailure("INVALID_OUTPUT", "output path must differ from input path")
    if not model_id:
        raise WorkerFailure("MODEL_NOT_FOUND", "model_id is required")

    requested_sample_rate = request.get("output_sample_rate")
    source_rate, channels, source_duration = probe_audio(input_path)
    sample_rate = int(requested_sample_rate or source_rate)
    if sample_rate <= 0 or channels <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "invalid audio sample rate or channel count")
    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    model_path = find_model(model_id, model_dir)
    device = choose_device(str(request.get("device", "auto")))
    emit_progress(request_id, "load_model", 5.0, message=f"loading {model_path.name}")
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
    except RuntimeError as error:
        raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error

    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")
    audio = decode_audio(input_path, sample_rate, channels)
    chunk_seconds = float(request.get("chunk_seconds", 20.0))
    overlap_seconds = float(request.get("overlap_seconds", 2.0))
    chunk_size = max(1, int(chunk_seconds * sample_rate))
    overlap = min(chunk_size - 1, max(0, int(overlap_seconds * sample_rate)))
    step = chunk_size - overlap
    output = np.zeros_like(audio)
    weights = np.zeros((1, audio.shape[1]), dtype=np.float32)
    total_seconds = audio.shape[1] / sample_rate
    starts = range(0, audio.shape[1], step)
    start_list = list(starts)

    for index, start in enumerate(start_list):
        if cancel_event.is_set():
            raise WorkerFailure("CANCELLED", "processing cancelled")
        end = min(audio.shape[1], start + chunk_size)
        actual = audio[:, start:end]
        padded = np.zeros((channels, chunk_size), dtype=np.float32)
        padded[:, : actual.shape[1]] = actual
        predicted = infer_chunk(model, padded, device)
        valid = actual.shape[1]
        blend = np.ones(valid, dtype=np.float32)
        if start > 0:
            blend[: min(overlap, valid)] = np.linspace(
                0.0, 1.0, min(overlap, valid), endpoint=False, dtype=np.float32
            )
        if end < audio.shape[1]:
            blend[-min(overlap, valid) :] *= np.linspace(
                1.0, 0.0, min(overlap, valid), endpoint=False, dtype=np.float32
            )
        output[:, start:end] += predicted[:, :valid] * blend
        weights[:, start:end] += blend
        emit_progress(
            request_id,
            "inference",
            10.0 + 80.0 * (index + 1) / max(1, len(start_list)),
            processed_seconds=end / sample_rate,
            total_seconds=total_seconds or source_duration,
        )

    output /= np.maximum(weights, 1e-8)
    if not np.isfinite(output).all():
        raise WorkerFailure("OUTPUT_INVALID", "enhanced output contains NaN or Inf")
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        output = output / peak
        peak = 1.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    emit_progress(request_id, "write_output", 95.0, message="writing AI master")
    encode_audio(output, output_path, sample_rate, channels)
    if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
        raise WorkerFailure("OUTPUT_INVALID", "AI output file is empty")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "result",
        "status": "completed",
        "output_path": output_path,
        "model_id": model_id,
        "model_version": model_path.stem,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": audio.shape[1] / sample_rate,
        "peak_db": float(20.0 * np.log10(max(peak, 1e-8))),
    }


def process_in_thread(request: dict[str, Any], cancel_event: threading.Event) -> None:
    request_id = str(request.get("request_id", ""))
    try:
        emit(enhance(request, cancel_event))
    except WorkerFailure as failure:
        emit_error(request_id, failure)
    except Exception as error:  # Keep unexpected model errors inside the protocol.
        print(f"unexpected worker error: {error!r}", file=sys.stderr, flush=True)
        emit_error(request_id, WorkerFailure("INFERENCE_FAILED", str(error)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    if args.model_dir:
        os.environ["AUDIO_AI_MODEL_DIR"] = args.model_dir

    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "",
            "type": "ready",
            "worker_version": "torchscript-0.1.0",
            "models": available_models(model_dir),
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
                emit_error(request_id, WorkerFailure("BUSY", "worker is already processing"))
                continue
            cancel_event.clear()
            active_thread = threading.Thread(
                target=process_in_thread, args=(request, cancel_event), daemon=True
            )
            active_thread.start()
        elif command == "cancel":
            cancel_event.set()
        elif command == "shutdown":
            cancel_event.set()
            if active_thread and active_thread.is_alive():
                active_thread.join(timeout=1.0)
            return 0
        else:
            emit_error(request_id, WorkerFailure("INVALID_COMMAND", f"unsupported command: {command}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
