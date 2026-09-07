"""FlashSR 推理流水线：探测、解码、定长分块推理、重叠相加、编码。

本模块在收到 `process` 命令之后才被 `worker.py` 导入（风险 R12）：顶层
`import torch` 实测约 5.7 秒，若放在启动时执行会超出 Rust 侧 3 秒的
ready 握手超时。清单解析留在 `worker.py`（纯标准库），这样 `from worker
import find_model` 不会拉起 torch。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import backend
from protocol import (
    PROTOCOL_VERSION,
    SCRIPT_ROOT,
    WorkerFailure,
    emit_progress,
    log,
    resolve_binary,
)

# worker.py 是纯标准库模块，这里导入它不会拉起 torch。
# 以脚本方式运行时 worker 是 __main__，本导入会额外产生一个 worker 模块
# 对象；两者共用同一个 protocol 模块，因此 WorkerFailure 仍是同一个类。
from worker import ModelSpec, find_model, ordered_weights

# 编码时每个写块的目标采样点数（约 1 MB / 块，兼顾吞吐与内存占用）
ENCODE_BLOCK_SAMPLES = 262_144

#: 进度区间：与 audio_ai 保持一致，前端无需区分后端
PERCENT_LOAD_MODEL = 5.0
PERCENT_PREPARE = 10.0
PERCENT_INFERENCE_START = 10.0
PERCENT_INFERENCE_SPAN = 80.0
PERCENT_WRITE_OUTPUT = 95.0


# --------------------------------------------------------------------------
# 输入探测与解码
# --------------------------------------------------------------------------


def probe_audio(path: str) -> tuple[int, int, float]:
    """用 ffprobe 读取采样率、声道数与时长。"""
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
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise WorkerFailure("UNSUPPORTED_INPUT", f"invalid ffprobe output: {error}") from error


def decode_audio(path: str, sample_rate: int, channels: int) -> np.ndarray:
    """把输入音频解码为 `(channels, frames)` 的 float32。

    `channels` 已在调用前限制为 1 或 2：上游模型按 `[B, T]` 处理，超过两声道
    未定义，多声道输入在此统一由 FFmpeg 下混。
    """
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


# --------------------------------------------------------------------------
# 输出编码
# --------------------------------------------------------------------------


def codec_for_output(output_path: str) -> str:
    suffix = output_path.lower()
    if suffix.endswith(".part"):
        suffix = suffix[:-5]
    if suffix.endswith(".flac"):
        return "flac"
    if suffix.endswith(".wav"):
        return "pcm_s16le"
    raise WorkerFailure("UNSUPPORTED_OUTPUT", "AI master output must be WAV or FLAC")


# 音频编码器 → 容器格式。Rust 侧先把结果写成 `xxx.flac.part` 再改名，
# `.part` 后缀让 ffmpeg 无法推断容器，因此必须显式传 `-f`。
_CONTAINER_FORMATS = {"flac": "flac", "pcm_s16le": "wav"}


def ffmpeg_encode_command(
    output_path: str, sample_rate: int, channels: int, codec: str
) -> list[str]:
    return [
        resolve_binary("ffmpeg", "AUDIO_AI_FFMPEG"),
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
        "-f",
        _CONTAINER_FORMATS.get(codec, codec),
        output_path,
    ]


def run_ffmpeg_with_blocks(command: list[str], blocks: Iterator[np.ndarray]) -> None:
    """把交错的 PCM 块（形状 (frames, channels) 的 float32）写入 ffmpeg。

    交错后的 PCM 可能高达数百 MB：一次性转成 bytes 会让峰值内存翻倍。这里
    分块写入 stdin，并用临时文件接收 stderr，避免双向管道互相阻塞。
    """
    try:
        with tempfile.TemporaryFile() as error_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=error_file,
            )
            try:
                stdin = process.stdin
                assert stdin is not None
                for block in blocks:
                    stdin.write(block.tobytes())
            except (BrokenPipeError, ValueError, OSError):
                # ffmpeg 提前退出（参数 / 路径问题）；真实原因在 stderr 中
                pass
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
            returncode = process.wait()
            error_file.seek(0)
            detail = error_file.read().decode("utf-8", errors="replace").strip()
    except OSError as error:
        raise WorkerFailure("ENCODE_FAILED", f"cannot start ffmpeg: {error}") from error
    if returncode != 0:
        raise WorkerFailure("ENCODE_FAILED", detail or "ffmpeg encode failed")


def encode_pcm_file(
    pcm_path: Path,
    output_path: str,
    sample_rate: int,
    channels: int,
    gain: float = 1.0,
) -> None:
    """流式把 f32le 暂存文件编码为目标格式，内存中只保留一个块。"""
    block_frames = max(1, ENCODE_BLOCK_SAMPLES // max(1, channels))
    frame_bytes = max(1, channels) * 4

    def blocks() -> Iterator[np.ndarray]:
        with open(pcm_path, "rb") as source:
            while True:
                raw = source.read(block_frames * frame_bytes)
                if not raw:
                    return
                block = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)
                if gain != 1.0:
                    block = (block * gain).astype(np.float32)
                yield block

    # `blocks()` 是惰性生成器，文件句柄开在生成器内部。若 ffmpeg 提前退出
    # 或循环提前结束，生成器不会被耗尽，`with open(...)` 就不会执行，句柄
    # 一直挂着 —— 随后删除暂存 PCM 会在 Windows 上报
    # PermissionError(WinError 32)「文件正被占用」（阶段 6 实测）。
    # 因此必须显式 close()，强迫生成器在 yield 处抛 GeneratorExit 并释放句柄。
    generator = blocks()
    try:
        run_ffmpeg_with_blocks(
            ffmpeg_encode_command(
                output_path, sample_rate, channels, codec_for_output(output_path)
            ),
            generator,
        )
    finally:
        generator.close()


# --------------------------------------------------------------------------
# 重叠相加与落盘
# --------------------------------------------------------------------------


class ChunkAssembler:
    """分块推理结果的重叠相加（OLA）累加器，支持定稿后立即输出。

    相邻分块之间有淡入淡出重叠，所以某段样本只有在「下一块处理完、不会再
    被继续累加」之后才算定稿。定稿后立即取出并写盘，完整结果就无需常驻
    内存 —— 长音频可省下数百 MB。

    语义与「先整体累加、最后统一除以权重」完全等价：归一化是逐元素运算，
    每个样本最终的累加值与累加权重都不变，提前做除法结果一致。
    """

    def __init__(self, channels: int) -> None:
        self._flush_pos = 0
        self._pending = np.zeros((channels, 0), dtype=np.float32)
        self._weights = np.zeros((1, 0), dtype=np.float32)

    def add(self, start: int, end: int, predicted: np.ndarray, blend: np.ndarray) -> None:
        """把 [start, end) 的推理结果按 blend 权重累加进待定稿区。"""
        valid = end - start
        if valid <= 0:
            return
        need = end - self._flush_pos
        if self._pending.shape[1] < need:
            grow = need - self._pending.shape[1]
            self._pending = np.pad(self._pending, ((0, 0), (0, grow)))
            self._weights = np.pad(self._weights, ((0, 0), (0, grow)))
        offset = start - self._flush_pos
        self._pending[:, offset : offset + valid] += predicted[:, :valid] * blend
        self._weights[:, offset : offset + valid] += blend

    def take_final(self, boundary: int) -> np.ndarray | None:
        """取出已定稿的 [已写出位置, boundary) 样本（已按权重归一化）。"""
        count = boundary - self._flush_pos
        if count <= 0:
            return None
        return self._pop(count)

    def remaining(self) -> np.ndarray | None:
        """取出全部剩余样本。"""
        if self._pending.shape[1] == 0:
            return None
        return self._pop(self._pending.shape[1])

    def _pop(self, count: int) -> np.ndarray:
        region = self._pending[:, :count] / np.maximum(self._weights[:, :count], 1e-8)
        self._pending = self._pending[:, count:]
        self._weights = self._weights[:, count:]
        self._flush_pos += count
        return np.ascontiguousarray(region, dtype=np.float32)


class PcmChunkWriter:
    """把定稿的 `(channels, frames)` 音频块顺序追加到 f32le 暂存文件。

    与 `ChunkAssembler` 配套：后者负责重叠相加，本类负责把定稿结果尽快
    落盘，因此完整结果无需常驻内存。

    同时增量统计峰值。样本一旦落盘就不再回头，峰值只能在这里统计；
    `audio_ai` 曾把 peak 初始化为 0.0 却从不更新，导致增益恒为 1.0、
    peak_db 恒为 −160 dB。
    """

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.peak = 0.0
        handle = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".f32le", prefix="audio-processor-pcm-", delete=False
        )
        self._handle = handle
        self.path = Path(handle.name)

    def write(self, region: np.ndarray | None) -> None:
        """写入一块定稿音频；`None` 表示当前没有可定稿的样本。"""
        if region is None or region.size == 0:
            return
        magnitude = float(np.abs(region).max())
        if magnitude > self.peak:
            self.peak = magnitude
        # 暂存文件是交错布局（frame-major），与 encode_pcm_file 的读取方式一致
        self._handle.write(np.ascontiguousarray(region.T, dtype=np.float32).tobytes())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def discard(self) -> None:
        """关闭并删除暂存文件。成功与失败路径都要调用，避免留下大文件。"""
        try:
            self.close()
        finally:
            self.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 设备与编排
# --------------------------------------------------------------------------


def warm_up_imports() -> None:
    """在主线程完成上游 `FlashSR` 的导入。

    `backend.import_flashsr()` 会经 `joblib → loky` 引入
    `concurrent.futures.process`，其导入会拉起 multiprocessing 的
    resource tracker 并与主线程交互。若在推理线程内导入，而主线程正阻塞于
    stdin 读取，会死锁 —— stdin 为管道时（Rust 侧的常态）必现：Worker 发出
    `load_model` 进度后再无任何事件，Rust 永远等不到 result（阶段 6 实测）。

    因此在主线程处理 `process` 命令时先预热，线程内只做纯计算。
    """
    backend.import_flashsr()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def enhance(request: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    """执行一次 FlashSR 超分，返回 `result` 事件载荷。"""
    request_id = str(request.get("request_id", ""))
    input_path = str(request.get("input_path", ""))
    output_path = str(request.get("output_path", ""))
    model_id = str(request.get("model_id", ""))

    if not input_path or not Path(input_path).is_file():
        # 回显实际收到的路径，便于定位「区域编码导致非 ASCII 路径被破坏」
        raise WorkerFailure("UNSUPPORTED_INPUT", f"input audio does not exist: {input_path!r}")
    if not output_path or Path(output_path).resolve() == Path(input_path).resolve():
        raise WorkerFailure("INVALID_OUTPUT", "output path must differ from input path")
    if not model_id:
        raise WorkerFailure("MODEL_NOT_FOUND", "model_id is required")

    source_rate, channels, source_duration = probe_audio(input_path)
    if source_rate <= 0 or channels <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "invalid audio sample rate or channel count")

    # 上游按 [B, T] 处理，第 0 维是 batch：超过两声道未定义，统一下混。
    downmixed = False
    if channels > backend.MAX_CHANNELS:
        log(f"输入为 {channels} 声道，超过 FlashSR 上限，下混为 {backend.MAX_CHANNELS} 声道")
        downmixed = True
        channels = backend.MAX_CHANNELS

    requested_sample_rate = request.get("output_sample_rate")
    if requested_sample_rate not in (None, backend.FLASHSR_SAMPLE_RATE):
        raise WorkerFailure(
            "UNSUPPORTED_SAMPLE_RATE",
            f"flashsr requires a {backend.FLASHSR_SAMPLE_RATE} Hz output sample rate",
        )
    sample_rate = backend.FLASHSR_SAMPLE_RATE

    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    spec: ModelSpec = find_model(model_id, model_dir)
    device = choose_device(str(request.get("device", "auto")))

    num_steps = int(request.get("num_steps", backend.DEFAULT_NUM_STEPS))
    if num_steps < 1:
        raise WorkerFailure("INVALID_REQUEST", "num_steps must be at least 1")
    lowpass_input = bool(request.get("lowpass_input", backend.DEFAULT_LOWPASS_INPUT))

    emit_progress(request_id, "load_model", PERCENT_LOAD_MODEL, message=f"loading {model_id}")
    weights = ordered_weights(spec, backend.WEIGHT_NAMES)
    model = backend.cached_flashsr(weights, device)

    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")

    audio = decode_audio(input_path, sample_rate, channels)
    frames = audio.shape[1]
    if frames <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "input audio contains no samples")
    if downmixed:
        emit_progress(
            request_id,
            "prepare_input",
            PERCENT_PREPARE,
            message=f"输入已下混为 {channels} 声道（FlashSR 最多支持 2 声道）",
        )

    chunk_samples = backend.FLASHSR_CHUNK_SAMPLES
    overlap_seconds = float(request.get("overlap_seconds", backend.DEFAULT_OVERLAP_SECONDS))
    if not (overlap_seconds >= 0.0):
        raise WorkerFailure("INVALID_REQUEST", "overlap_seconds must be non-negative")
    overlap = min(chunk_samples - 1, int(overlap_seconds * sample_rate))
    step = chunk_samples - overlap

    assembler = ChunkAssembler(channels)
    sink = PcmChunkWriter(channels)
    total_seconds = frames / sample_rate
    start_list = list(range(0, frames, step))

    try:
        for index, start in enumerate(start_list):
            if cancel_event.is_set():
                raise WorkerFailure("CANCELLED", "processing cancelled")
            end = min(frames, start + chunk_samples)
            valid = end - start
            # 尾块不足定长时补零：模型硬限制 245760 样本，不接受更短的输入
            block = np.zeros((channels, chunk_samples), dtype=np.float32)
            block[:, :valid] = audio[:, start:end]
            predicted = backend.infer_flashsr(
                model, block, device, num_steps=num_steps, lowpass_input=lowpass_input
            )
            predicted = predicted[:, :valid]

            blend = np.ones(valid, dtype=np.float32)
            # overlap 为 0 时必须整段跳过：blend[-0:] 等于 blend[0:]（整个
            # 数组），与长度为 0 的 linspace 相乘会抛 broadcast 错误。
            fade = min(overlap, valid)
            if fade > 0:
                if start > 0:
                    blend[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
                if end < frames:
                    blend[-fade:] *= np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
            assembler.add(start, end, predicted, blend)

            # [start, 下一块起点) 不会再被后续分块修改，已定稿 → 立即写盘
            next_start = start_list[index + 1] if index + 1 < len(start_list) else frames
            sink.write(assembler.take_final(next_start))
            emit_progress(
                request_id,
                "inference",
                PERCENT_INFERENCE_START
                + PERCENT_INFERENCE_SPAN * (index + 1) / len(start_list),
                processed_seconds=end / sample_rate,
                total_seconds=total_seconds or source_duration,
            )

        # 收尾：写出最后一段剩余样本
        sink.write(assembler.remaining())
        sink.close()

        peak = sink.peak
        if not np.isfinite(peak) or peak < 0.0:
            raise WorkerFailure("OUTPUT_INVALID", "enhanced output contains NaN or Inf")
        gain = 1.0 / peak if peak > 1.0 else 1.0
        if peak > 1.0:
            peak = 1.0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        emit_progress(request_id, "write_output", PERCENT_WRITE_OUTPUT, message="writing AI master")
        log(
            f"encode start: output={output_path} rate={sample_rate} "
            f"channels={channels} frames={frames} gain={gain:.6f} peak={peak:.6f}"
        )
        encode_pcm_file(sink.path, output_path, sample_rate, channels, gain=gain)
        written = Path(output_path).stat().st_size if Path(output_path).is_file() else -1
        log(f"encode done: size={written}")
        if not Path(output_path).is_file() or written == 0:
            raise WorkerFailure("OUTPUT_INVALID", "AI output file is empty")
        log("encode verified, building result payload")
    finally:
        # 成功、失败、取消三条路径都要清理：暂存 PCM 长音频可达数百 MB
        sink.discard()

    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "result",
        "status": "completed",
        "output_path": output_path,
        "model_id": model_id,
        "model_version": spec.version,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": frames / sample_rate,
        "peak_db": float(20.0 * np.log10(max(peak, 1e-8))),
    }
