"""Python audio enhancement worker with explicit local model backends.

TorchScript models receive a float32 tensor shaped [1, channels, samples].
The manifest can instead select native Python backends such as DeepFilterNet
or AudioSR. The worker must be supplied model files separately; it does not
download weights and never treats resampling as AI enhancement.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any

# AudioSR indirectly imports Transformers. Keep the inference Worker offline;
# model installation is handled by the separate, explicit model_manager.py.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch


PROTOCOL_VERSION = 1
SCRIPT_ROOT = Path(__file__).resolve().parents[2]
EMIT_LOCK = threading.Lock()
DEEPFILTER_CACHE_LOCK = threading.Lock()
DEEPFILTER_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
AUDIOSR_CACHE_LOCK = threading.Lock()
AUDIOSR_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}

# AudioSR warns about degraded quality for inputs longer than this. Keeping
# inference windows within the model's intended duration also bounds memory.
AUDIOSR_MAX_CHUNK_SECONDS = 10.24
AUDIOSR_DDIM_STEPS = 50
AUDIOSR_GUIDANCE_SCALE = 3.5
STARTUP_HASH_MAX_BYTES = 128 * 1024 * 1024


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


def read_manifest(model_dir: Path) -> dict[str, dict[str, str]] | None:
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
        result = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("model entry must be an object")
            model_id = str(entry["id"])
            backend = str(entry.get("backend", "torchscript"))
            if backend not in {"torchscript", "deepfilternet", "audiosr"}:
                raise ValueError(f"unsupported model backend: {backend}")
            parsed = {
                "file": str(entry["file"]),
                "version": str(entry.get("version", "unknown")),
                "sha256": str(entry["sha256"]).lower(),
                "backend": backend,
                "runtime_path": str(entry.get("runtime_path", "")),
                "model_name": str(entry.get("model_name", "")),
                "size_bytes": str(entry.get("size_bytes", "")),
            }
            if backend == "audiosr" and not parsed["model_name"]:
                raise ValueError("audiosr model_name is required")
            if parsed["size_bytes"] and int(parsed["size_bytes"]) <= 0:
                raise ValueError("size_bytes must be a positive integer")
            result[model_id] = parsed
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise WorkerFailure("MODEL_MANIFEST_INVALID", str(error)) from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_model(model_id: str, model_dir: Path) -> tuple[Path, str, str, Path, str]:
    manifest = read_manifest(model_dir)
    if manifest is not None:
        entry = manifest.get(model_id)
        if entry is None:
            raise WorkerFailure("MODEL_NOT_FOUND", f"model is not in manifest: {model_id}")
        candidate = (model_dir / entry["file"]).resolve()
        if model_dir.resolve() not in candidate.parents:
            raise WorkerFailure("MODEL_MANIFEST_INVALID", "model file escapes model directory")
        if not candidate.is_file():
            raise WorkerFailure("MODEL_NOT_FOUND", f"model file does not exist: {candidate.name}")
        expected_size = entry["size_bytes"]
        if expected_size and candidate.stat().st_size != int(expected_size):
            raise WorkerFailure("MODEL_SIZE_MISMATCH", f"model size mismatch: {candidate.name}")
        actual = sha256_file(candidate)
        if actual != entry["sha256"]:
            raise WorkerFailure("MODEL_HASH_MISMATCH", f"model hash mismatch: {candidate.name}")
        backend = entry["backend"]
        runtime_path = candidate
        if backend == "deepfilternet":
            configured_runtime = entry["runtime_path"]
            if not configured_runtime:
                raise WorkerFailure("MODEL_MANIFEST_INVALID", "deepfilternet runtime_path is required")
            runtime_path = (model_dir / configured_runtime).resolve()
            if model_dir.resolve() not in runtime_path.parents:
                raise WorkerFailure("MODEL_MANIFEST_INVALID", "model runtime path escapes model directory")
            if not runtime_path.is_dir():
                raise WorkerFailure(
                    "MODEL_NOT_FOUND", f"model runtime directory does not exist: {runtime_path}"
                )
        return candidate, entry["version"], backend, runtime_path, entry["model_name"]

    configured = os.environ.get("AUDIO_AI_MODEL")
    candidates = [Path(configured)] if configured else model_candidates(model_id, model_dir)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.stem, "torchscript", candidate, ""
    raise WorkerFailure("MODEL_NOT_FOUND", f"no TorchScript model found for {model_id}")


def available_models(model_dir: Path) -> tuple[list[str], list[str]]:
    manifest = read_manifest(model_dir)
    if manifest is not None:
        models = []
        errors = []
        for model_id, entry in manifest.items():
            candidate = (model_dir / entry["file"]).resolve()
            if model_dir.resolve() not in candidate.parents or not candidate.is_file():
                errors.append(f"{model_id}: MODEL_NOT_FOUND")
                continue
            expected_size = entry["size_bytes"]
            if expected_size and candidate.stat().st_size != int(expected_size):
                errors.append(f"{model_id}: MODEL_SIZE_MISMATCH")
                continue
            if candidate.stat().st_size > STARTUP_HASH_MAX_BYTES:
                # A multi-gigabyte SHA-256 before ready would exceed Rust's
                # short startup timeout. The same checksum is always verified
                # in find_model immediately before the model is loaded.
                models.append(model_id)
                errors.append(f"{model_id}: MODEL_HASH_DEFERRED")
                continue
            if sha256_file(candidate) != entry["sha256"]:
                errors.append(f"{model_id}: MODEL_HASH_MISMATCH")
                continue
            if entry["backend"] == "deepfilternet":
                runtime_path = entry["runtime_path"]
                if not runtime_path:
                    errors.append(f"{model_id}: MODEL_MANIFEST_INVALID")
                    continue
                runtime = (model_dir / runtime_path).resolve()
                if model_dir.resolve() not in runtime.parents or not runtime.is_dir():
                    errors.append(f"{model_id}: MODEL_NOT_FOUND")
                    continue
            models.append(model_id)
        return sorted(models), sorted(errors)
    if not model_dir.is_dir():
        return [], []
    names = set()
    for path in model_dir.iterdir():
        if path.suffix.lower() in {".ts", ".pt", ".pth"}:
            names.add(path.stem)
    return sorted(names), []


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def load_deepfilternet(model_dir: Path, device: torch.device) -> tuple[Any, Any, Any]:
    """Load DeepFilterNet without relying on its removed torchaudio I/O API."""
    if os.environ.get("AUDIO_AI_DEBUG") == "1":
        print(f"deepfilternet import: {model_dir}", file=sys.stderr, flush=True)
    try:
        import torchaudio
    except ImportError as error:
        raise WorkerFailure("MODEL_RUNTIME_NOT_FOUND", "deepfilternet requires torchaudio") from error

    if not hasattr(torchaudio, "info") and "df.io" not in sys.modules:
        # DeepFilterNet 0.5.6 imports these symbols for its CLI. The worker
        # already decodes and encodes through FFmpeg, so only import-time
        # compatibility is needed with torchaudio 2.11+.
        io_module = types.ModuleType("df.io")
        io_module.AudioMetaData = object
        io_module.load_audio = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("audio loading is handled by the AI worker")
        )
        io_module.resample = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("resampling is handled by the AI worker")
        )
        io_module.save_audio = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("audio writing is handled by the AI worker")
        )
        sys.modules["df.io"] = io_module

    try:
        from df.config import config
        from df.enhance import enhance as deepfilter_enhance
        from df.enhance import init_df
    except (ImportError, ModuleNotFoundError) as error:
        raise WorkerFailure(
            "MODEL_RUNTIME_NOT_FOUND", f"cannot import deepfilternet: {error}"
        ) from error

    try:
        model, df_state, _ = init_df(
            str(model_dir),
            log_level="ERROR",
            log_file=None,
            config_allow_defaults=True,
        )
        config.set("DEVICE", str(device), str, section="train")
        model = model.to(device).eval()
        if os.environ.get("AUDIO_AI_DEBUG") == "1":
            print("deepfilternet loaded", file=sys.stderr, flush=True)
        return model, df_state, deepfilter_enhance
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WorkerFailure("OUT_OF_MEMORY", str(error)) from error
        raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error
    except (OSError, ValueError, KeyError) as error:
        raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error


def cached_deepfilternet(model_dir: Path, device: torch.device) -> tuple[Any, Any, Any]:
    key = (str(model_dir.resolve()), str(device))
    with DEEPFILTER_CACHE_LOCK:
        cached = DEEPFILTER_CACHE.get(key)
        if cached is None:
            cached = load_deepfilternet(model_dir, device)
            DEEPFILTER_CACHE[key] = cached
        return cached


def load_audiosr(
    checkpoint_path: Path, model_name: str, device: torch.device
) -> tuple[Any, Any]:
    """Load AudioSR without allowing its helper to fetch a checkpoint."""
    if os.environ.get("AUDIO_AI_DEBUG") == "1":
        print(f"audiosr import: {checkpoint_path}", file=sys.stderr, flush=True)
    offline_patches = patch_audiosr_offline_dependencies()
    try:
        from audiosr import pipeline as audiosr_pipeline
    except Exception as error:  # AudioSR import can fail on incompatible native deps.
        restore_audiosr_offline_dependencies(offline_patches)
        raise WorkerFailure(
            "MODEL_RUNTIME_NOT_FOUND", f"cannot import audiosr: {error}"
        ) from error

    original_download_checkpoint = audiosr_pipeline.download_checkpoint

    def local_checkpoint(requested_model_name: str) -> str:
        if requested_model_name != model_name:
            raise RuntimeError(
                f"AudioSR requested unexpected model: {requested_model_name}"
            )
        return str(checkpoint_path)

    try:
        # AudioSR 0.0.7 ignores build_model(ckpt_path=...) and unconditionally
        # calls its module-level downloader. Patch only for this load and restore
        # it immediately, so an enhancement request can never fetch weights.
        audiosr_pipeline.download_checkpoint = local_checkpoint
        with contextlib.redirect_stdout(sys.stderr):
            model = audiosr_pipeline.build_model(
                device=device, model_name=model_name
            )
        if os.environ.get("AUDIO_AI_DEBUG") == "1":
            print("audiosr loaded", file=sys.stderr, flush=True)
        return model, audiosr_pipeline.super_resolution
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WorkerFailure("OUT_OF_MEMORY", str(error)) from error
        raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error
    except (OSError, ValueError, KeyError) as error:
        raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error
    finally:
        audiosr_pipeline.download_checkpoint = original_download_checkpoint
        restore_audiosr_offline_dependencies(offline_patches)


class OfflineRobertaTokenizer:
    """Enough of the tokenizer API for AudioSR's unconditional empty prompt."""

    def __call__(
        self,
        texts: Any,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        if isinstance(texts, str):
            texts = [texts]
        count = len(texts)
        input_ids = torch.full((count, max_length), 1, dtype=torch.long)
        attention_mask = torch.zeros((count, max_length), dtype=torch.long)
        token_type_ids = torch.zeros((count, max_length), dtype=torch.long)
        if max_length > 0:
            input_ids[:, 0] = 0
            attention_mask[:, 0] = 1
        if max_length > 1:
            input_ids[:, 1] = 2
            attention_mask[:, 1] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


def patch_audiosr_offline_dependencies() -> list[tuple[type[Any], str, bool, Any]]:
    """Patch AudioSR's unused text side so model loading never contacts Hub.

    AudioSR's unconditional audio conditioning does not use text, but version
    0.0.7 still constructs a Roberta tokenizer/config through Transformers.
    Those calls trigger network retries when the optional Hub files are absent.
    The model checkpoint contains the CLAP/audio weights; an empty-token batch
    only needs the standard Roberta token IDs and config shape during startup.
    """
    try:
        from transformers import RobertaConfig, RobertaTokenizer
    except Exception as error:
        raise WorkerFailure(
            "MODEL_RUNTIME_NOT_FOUND", f"cannot prepare AudioSR offline runtime: {error}"
        ) from error

    patches: list[tuple[type[Any], str, bool, Any]] = []
    for target, factory in (
        (RobertaTokenizer, lambda cls, *args, **kwargs: OfflineRobertaTokenizer()),
        (RobertaConfig, lambda cls, *args, **kwargs: cls()),
    ):
        had_own_factory = "from_pretrained" in target.__dict__
        original_factory = target.__dict__.get("from_pretrained")
        patches.append((target, "from_pretrained", had_own_factory, original_factory))
        target.from_pretrained = classmethod(factory)
    return patches


def restore_audiosr_offline_dependencies(
    patches: list[tuple[type[Any], str, bool, Any]]
) -> None:
    for target, name, had_own_factory, original_factory in patches:
        if had_own_factory:
            setattr(target, name, original_factory)
        else:
            delattr(target, name)


def cached_audiosr(
    checkpoint_path: Path, model_name: str, device: torch.device
) -> tuple[Any, Any]:
    key = (str(checkpoint_path.resolve()), model_name, str(device))
    with AUDIOSR_CACHE_LOCK:
        cached = AUDIOSR_CACHE.get(key)
        if cached is None:
            cached = load_audiosr(checkpoint_path, model_name, device)
            AUDIOSR_CACHE[key] = cached
        return cached


def infer_audiosr(
    model: Any,
    super_resolution_fn: Any,
    chunk: np.ndarray,
    temporary_dir: Path,
    chunk_index: int,
    channel_index: int,
) -> np.ndarray:
    """Run AudioSR on one mono chunk and trim its internal duration padding."""
    if chunk.shape[0] != 1:
        raise WorkerFailure("INFERENCE_FAILED", "audiosr input must be mono")
    input_path = temporary_dir / f"chunk-{chunk_index:05d}-channel-{channel_index}.wav"
    encode_audio(chunk, str(input_path), 48_000, 1)
    try:
        # AudioSR and its dependencies may print progress/logs. stdout belongs
        # exclusively to this worker's JSONL protocol.
        with contextlib.redirect_stdout(sys.stderr):
            predicted = super_resolution_fn(
                model,
                str(input_path),
                seed=42,
                ddim_steps=AUDIOSR_DDIM_STEPS,
                guidance_scale=AUDIOSR_GUIDANCE_SCALE,
            )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WorkerFailure("OUT_OF_MEMORY", str(error)) from error
        raise WorkerFailure("INFERENCE_FAILED", str(error)) from error
    except (OSError, ValueError) as error:
        raise WorkerFailure("INFERENCE_FAILED", str(error)) from error

    if isinstance(predicted, torch.Tensor):
        result = predicted.detach().float().cpu().numpy()
    else:
        result = np.asarray(predicted, dtype=np.float32)
    if result.ndim == 3 and result.shape[:2] == (1, 1):
        result = result[0, 0]
    elif result.ndim == 2 and result.shape[0] == 1:
        result = result[0]
    elif result.ndim != 1:
        raise WorkerFailure(
            "INFERENCE_FAILED",
            f"audiosr output has unsupported shape {result.shape}",
        )
    if result.size < chunk.shape[1]:
        raise WorkerFailure(
            "OUTPUT_INVALID",
            "audiosr output is shorter than its input chunk",
        )
    result = np.ascontiguousarray(result[: chunk.shape[1]], dtype=np.float32)
    if not np.isfinite(result).all():
        raise WorkerFailure("OUTPUT_INVALID", "audiosr output contains NaN or Inf")
    return result.reshape(1, -1)


def infer_deepfilternet(
    model: Any,
    df_state: Any,
    enhance_fn: Any,
    chunk: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    # DeepFilterNet computes FFT features from CPU NumPy audio and moves only
    # model features to the selected device.
    tensor = torch.from_numpy(np.ascontiguousarray(chunk)).float()
    try:
        with torch.inference_mode():
            output = enhance_fn(model, df_state, tensor, pad=True)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise WorkerFailure("OUT_OF_MEMORY", str(error)) from error
        raise WorkerFailure("INFERENCE_FAILED", str(error)) from error
    if not isinstance(output, torch.Tensor):
        raise WorkerFailure("INFERENCE_FAILED", "deepfilternet output is not a tensor")
    result = output.detach().float().cpu().numpy()
    if result.shape != chunk.shape:
        raise WorkerFailure(
            "INFERENCE_FAILED",
            f"model output shape {result.shape} does not match input {chunk.shape}",
        )
    if not np.isfinite(result).all():
        raise WorkerFailure("OUTPUT_INVALID", "model output contains NaN or Inf")
    return result


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
    model_path, model_version, backend, runtime_path, model_name = find_model(
        model_id, model_dir
    )
    if backend in {"deepfilternet", "audiosr"}:
        if requested_sample_rate not in (None, 48_000):
            raise WorkerFailure(
                "UNSUPPORTED_SAMPLE_RATE",
                f"{backend} requires a 48000 Hz output sample rate",
            )
        sample_rate = 48_000
    device = choose_device(str(request.get("device", "auto")))
    emit_progress(request_id, "load_model", 5.0, message=f"loading {model_path.name}")
    if backend == "torchscript":
        try:
            model = torch.jit.load(str(runtime_path), map_location=device)
            model.eval()
        except RuntimeError as error:
            raise WorkerFailure("MODEL_LOAD_FAILED", str(error)) from error
        model_state = None
        model_infer = None
    elif backend == "deepfilternet":
        model, model_state, model_infer = cached_deepfilternet(runtime_path, device)
    else:
        model, model_infer = cached_audiosr(model_path, model_name, device)
        model_state = None

    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")
    if os.environ.get("AUDIO_AI_DEBUG") == "1":
        print("decode audio", file=sys.stderr, flush=True)
    audio = decode_audio(input_path, sample_rate, channels)
    chunk_seconds = float(request.get("chunk_seconds", 20.0))
    overlap_seconds = float(request.get("overlap_seconds", 2.0))
    if backend == "audiosr" and chunk_seconds > AUDIOSR_MAX_CHUNK_SECONDS:
        chunk_seconds = AUDIOSR_MAX_CHUNK_SECONDS
        emit_progress(
            request_id,
            "prepare_input",
            8.0,
            message=(
                "AudioSR chunks are limited to "
                f"{AUDIOSR_MAX_CHUNK_SECONDS:.2f} seconds"
            ),
        )
    chunk_size = max(1, int(chunk_seconds * sample_rate))
    overlap = min(chunk_size - 1, max(0, int(overlap_seconds * sample_rate)))
    step = chunk_size - overlap
    output = np.zeros_like(audio)
    weights = np.zeros((1, audio.shape[1]), dtype=np.float32)
    total_seconds = audio.shape[1] / sample_rate
    starts = range(0, audio.shape[1], step)
    start_list = list(starts)

    temporary_dir_context = (
        tempfile.TemporaryDirectory(prefix="audio-processor-audiosr-")
        if backend == "audiosr"
        else None
    )
    try:
        for index, start in enumerate(start_list):
            if cancel_event.is_set():
                raise WorkerFailure("CANCELLED", "processing cancelled")
            end = min(audio.shape[1], start + chunk_size)
            actual = audio[:, start:end]
            if backend == "audiosr":
                predicted = np.zeros_like(actual)
                temporary_dir = Path(temporary_dir_context.name)
                for channel_index in range(channels):
                    if cancel_event.is_set():
                        raise WorkerFailure("CANCELLED", "processing cancelled")
                    predicted[channel_index : channel_index + 1] = infer_audiosr(
                        model,
                        model_infer,
                        actual[channel_index : channel_index + 1],
                        temporary_dir,
                        index,
                        channel_index,
                    )
            else:
                padded = np.zeros((channels, chunk_size), dtype=np.float32)
                padded[:, : actual.shape[1]] = actual
                if backend == "torchscript":
                    predicted = infer_chunk(model, padded, device)
                else:
                    if os.environ.get("AUDIO_AI_DEBUG") == "1":
                        print(
                            f"infer chunk {index + 1}/{len(start_list)}",
                            file=sys.stderr,
                            flush=True,
                        )
                    predicted = infer_deepfilternet(
                        model, model_state, model_infer, padded, device
                    )
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
    finally:
        if temporary_dir_context is not None:
            temporary_dir_context.cleanup()

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
        "model_version": model_version,
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
    try:
        models, model_errors = available_models(model_dir)
    except WorkerFailure as failure:
        models, model_errors = [], [f"{failure.code}: {failure}"]
    emit(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "",
            "type": "ready",
            "worker_version": "python-ai-0.3.0",
            "models": models,
            "model_errors": model_errors,
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
            # DeepFilterNet initializes global device/logging state. Prepare it
            # on the protocol thread, then let the worker thread reuse the cache.
            try:
                model_dir = Path(
                    os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models"))
                )
                _, _, backend, runtime_path, _ = find_model(
                    str(request.get("model_id", "")), model_dir
                )
                if backend == "deepfilternet":
                    cached_deepfilternet(
                        runtime_path, choose_device(str(request.get("device", "auto")))
                    )
            except WorkerFailure as failure:
                emit_error(request_id, failure)
                continue
            except Exception as error:
                emit_error(request_id, WorkerFailure("MODEL_LOAD_FAILED", str(error)))
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
