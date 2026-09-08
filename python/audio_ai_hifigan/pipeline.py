"""HiFi-GAN 推理流水线：探测、解码、逐声道 mel 提取、生成器前向、编码。

本模块在收到 `process` 命令之后才被 `worker.py` 导入（风险 R12）：顶层
`import torch` 实测约 5.7 秒，若放在启动时执行会超出 Rust 侧 3 秒的
ready 握手超时。清单解析留在 `worker.py`（纯标准库），这样 `from worker
import find_model` 不会拉起 torch。

HiFi-GAN 是**声码器**（mel → waveform），与 FlashSR 的扩散超分不同：它没有
固定尺寸输入限制，可整段前向。本流水线对每个声道独立「提取 mel → 生成器
前向 → 波形」，再按原序合并（R8：生成器单声道）。整段前向的边界连续性天然
优于分块，因此不做重叠相加；长音频的内存由 `PcmChunkWriter` 流式落盘。
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

import vendor_bridge
from protocol import (
    PROTOCOL_VERSION,
    SCRIPT_ROOT,
    WorkerFailure,
    emit_progress,
    log,
    resolve_binary,
)
from worker import ModelSpec, find_model, ordered_weights

#: HiFi-GAN 权重文件名（manifest `files[].name`）。
MODEL_WEIGHT_NAME = ("generator",)

# 编码时每个写块的目标采样点数（约 1 MB / 块，兼顾吞吐与内存占用）
ENCODE_BLOCK_SAMPLES = 262_144

#: 进度区间：与 audio_ai / FlashSR 保持一致，前端无需区分后端
PERCENT_LOAD_MODEL = 5.0
PERCENT_PREPARE = 10.0
PERCENT_INFERENCE_START = 10.0
PERCENT_INFERENCE_SPAN = 80.0
PERCENT_WRITE_OUTPUT = 95.0

#: 分块推理。HiFi-GAN 是全卷积、理论上可整段前向，但整段跑有两个实际问题：
#: ① 进度只在「一个声道跑完」后才推进，长音频会长时间停在 10%，表现为卡死；
#: ② 中间张量随时长线性增长，长音频内存峰值可达数 GB。
#: Rust 侧按后端下发 `chunk_seconds`（HiFi-GAN 默认 30s），此处据此切块，
#: 与 FlashSR「Python 侧自行切块」的既有约定一致。
DEFAULT_CHUNK_SECONDS = 30.0

#: 块两侧的上下文余量（秒）。卷积感受野在块边界会产生伪影，故每块多取一段
#: 作为上下文，生成后按原区间裁掉，避免拼接处出现爆音。
CONTEXT_PAD_SECONDS = 0.5

#: 运行时缓存：同一 (权重路径, 设备) 只加载一次生成器。
_RUNNER_CACHE: dict[tuple[str, str], Any] = {}
_RUNNER_LOCK = threading.Lock()


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

    `channels` 已在调用前限制为 1 或 2：生成器为单声道，逐声道推理；多于两声道
    在调用前统一由 FFmpeg 下混。
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
    # 声码器只重建 mel 已含的频率：削波到 [-1, 1] 避免极端样本破坏 mel 提取
    return np.clip(audio, -1.0, 1.0)


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


class PcmChunkWriter:
    """把 `(channels, frames)` 音频块顺序追加到 f32le 暂存文件，并增量统计峰值。"""

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
    """在主线程完成 vendor 与生成器类的导入，避免推理线程内死锁。

    与 FlashSR 同因：vendor 模块经 joblib → loky 引入 multiprocessing 的
    resource tracker，若在线程内导入而主线程阻塞于 stdin 读取会死锁。
    这里仅做导入与路径引导（不加载权重），权重在 `enhance` 内按模型加载。
    """
    vendor_bridge.bootstrap_vendor_path()
    vendor_bridge.ensure_inference_only_imports()
    # 触发生成器类的解析，确保主线程已完成潜在重导入
    vendor_bridge.build_generator(remove_weight_norm=True)
    # mel 提取器同样必须在主线程完成导入与构造：`UtilAudioMelSpec` 的导入链经
    # joblib → loky 拉起 multiprocessing 的 resource tracker，若留到推理线程内
    # 首次导入，而主线程正阻塞于 stdin 读取，二者死锁（stdin 为管道时必现）。
    # 这正是「短音频也卡在 10%」的根因 —— 10% 是 prepare_input，下一步就是首次
    # mel 提取。仅构造提取器，不加载权重。
    vendor_bridge.warm_up_mel_extractor()
    # 重采样用到的 scipy.signal 一并预热，避免同样的线程内首次导入风险。
    import scipy.signal  # noqa: F401


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def get_runner(generator_path: Path, device: torch.device, config: dict[str, Any]) -> Any:
    """按 (权重路径, 设备) 缓存已加载的生成器，避免重复加载。"""
    key = (str(generator_path), str(device))
    with _RUNNER_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = vendor_bridge.HiFiGanRunner(config=config, device=device)
            runner.load_model(str(generator_path))
            _RUNNER_CACHE[key] = runner
        return runner


def enhance(request: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    """执行一次 HiFi-GAN 声码器重建，返回 `result` 事件载荷。"""
    request_id = str(request.get("request_id", ""))
    input_path = str(request.get("input_path", ""))
    output_path = str(request.get("output_path", ""))
    model_id = str(request.get("model_id", ""))

    if not input_path or not Path(input_path).is_file():
        raise WorkerFailure("UNSUPPORTED_INPUT", f"input audio does not exist: {input_path!r}")
    if not output_path or Path(output_path).resolve() == Path(input_path).resolve():
        raise WorkerFailure("INVALID_OUTPUT", "output path must differ from input path")
    if not model_id:
        raise WorkerFailure("MODEL_NOT_FOUND", "model_id is required")

    source_rate, channels, source_duration = probe_audio(input_path)
    if source_rate <= 0 or channels <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "invalid audio sample rate or channel count")

    # 生成器为单声道：超过两声道未定义，统一下混为 2。
    downmixed = False
    if channels > 2:
        log(f"输入为 {channels} 声道，超过 HiFi-GAN 上限，下混为 2 声道")
        downmixed = True
        channels = 2

    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    spec: ModelSpec = find_model(model_id, model_dir)
    device = choose_device(str(request.get("device", "auto")))

    config = vendor_bridge.get_config(spec.native_sample_rate)
    native_rate = config["sampling_rate"]
    # 输出固定为 48k（与 App 其它后端对齐）；原生 22.05k 权重在下游重采样到 48k。
    TARGET_RATE = 48000

    requested_sample_rate = request.get("output_sample_rate")
    if requested_sample_rate not in (None, TARGET_RATE):
        raise WorkerFailure(
            "UNSUPPORTED_SAMPLE_RATE",
            f"hifigan requires a {TARGET_RATE} Hz output sample rate",
        )

    emit_progress(request_id, "load_model", PERCENT_LOAD_MODEL, message=f"loading {model_id}")
    generator_path = ordered_weights(spec, MODEL_WEIGHT_NAME)[0]
    runner = get_runner(generator_path, device, config)

    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")

    audio = decode_audio(input_path, native_rate, channels)
    frames = audio.shape[1]
    if frames <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "input audio contains no samples")
    if downmixed:
        emit_progress(
            request_id,
            "prepare_input",
            PERCENT_PREPARE,
            message=f"输入已下混为 {channels} 声道（HiFi-GAN 最多支持 2 声道）",
        )
    else:
        emit_progress(request_id, "prepare_input", PERCENT_PREPARE, message="解码完成，提取 mel")

    total_seconds = frames / native_rate
    chunk_seconds = request.get("chunk_seconds")
    try:
        chunk_seconds = float(chunk_seconds) if chunk_seconds is not None else 0.0
    except (TypeError, ValueError):
        chunk_seconds = 0.0
    if not (chunk_seconds > 0):
        chunk_seconds = DEFAULT_CHUNK_SECONDS
    chunk_frames = max(1, int(chunk_seconds * native_rate))
    pad_frames = min(chunk_frames, int(CONTEXT_PAD_SECONDS * native_rate))

    predicted_channels: list[np.ndarray] = []
    for ch in range(audio.shape[0]):
        if cancel_event.is_set():
            raise WorkerFailure("CANCELLED", "processing cancelled")
        channel_audio = audio[ch]
        starts = list(range(0, frames, chunk_frames))
        pieces: list[np.ndarray] = []
        for index, start in enumerate(starts):
            if cancel_event.is_set():
                raise WorkerFailure("CANCELLED", "processing cancelled")
            end = min(frames, start + chunk_frames)
            # 多取 pad 作为上下文，生成后裁掉，消除块边界的感受野伪影
            seg_start = max(0, start - pad_frames)
            seg_end = min(frames, end + pad_frames)
            segment = channel_audio[seg_start:seg_end]
            mel = runner.audio_to_mel(segment).squeeze()
            wav = runner.mel_to_audio(mel)
            # 生成长度与输入段长度可能有一帧级偏差，按比例换算裁剪位置
            scale = wav.shape[0] / segment.shape[0] if segment.shape[0] else 1.0
            left = int(round((start - seg_start) * scale))
            right = min(wav.shape[0], int(round((end - seg_start) * scale)))
            pieces.append(wav[left:right])
            done = ch + (index + 1) / len(starts)
            emit_progress(
                request_id,
                "inference",
                PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN * done / audio.shape[0],
                processed_seconds=total_seconds * done / audio.shape[0],
                total_seconds=total_seconds or source_duration,
            )
        predicted_channels.append(np.concatenate(pieces))

    predicted = np.stack(predicted_channels, axis=0)
    if not np.isfinite(predicted).all():
        raise WorkerFailure("OUTPUT_INVALID", "reconstructed output contains NaN or Inf")

    # 若权重原生采样率不是 48k（如 22.05k），重建后重采样到目标 48k（计划 §3.4）。
    if native_rate != TARGET_RATE:
        from scipy.signal import resample_poly

        resampled = [
            resample_poly(predicted[ch], TARGET_RATE, native_rate)
            for ch in range(predicted.shape[0])
        ]
        predicted = np.stack(resampled, axis=0)
        emit_progress(
            request_id,
            "resample",
            (PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN + PERCENT_WRITE_OUTPUT) / 2,
            message=f"重采样 {native_rate}→{TARGET_RATE} Hz",
        )

    sink = PcmChunkWriter(channels)
    sink.write(predicted)
    sink.close()

    peak = sink.peak
    if not np.isfinite(peak) or peak < 0.0:
        sink.discard()
        raise WorkerFailure("OUTPUT_INVALID", "reconstructed output contains NaN or Inf")
    gain = 1.0 / peak if peak > 1.0 else 1.0
    if peak > 1.0:
        peak = 1.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    emit_progress(request_id, "write_output", PERCENT_WRITE_OUTPUT, message="writing AI master")
    log(
        f"encode start: output={output_path} rate={TARGET_RATE} "
        f"channels={channels} gain={gain:.6f} peak={peak:.6f}"
    )
    try:
        encode_pcm_file(sink.path, output_path, TARGET_RATE, channels, gain=gain)
    finally:
        sink.discard()
    written = Path(output_path).stat().st_size if Path(output_path).is_file() else -1
    log(f"encode done: size={written}")
    if not Path(output_path).is_file() or written == 0:
        raise WorkerFailure("OUTPUT_INVALID", "AI output file is empty")
    log("encode verified, building result payload")

    out_frames = predicted.shape[1]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "type": "result",
        "status": "completed",
        "output_path": output_path,
        "model_id": model_id,
        "model_version": spec.version,
        "sample_rate": TARGET_RATE,
        "channels": channels,
        "duration_seconds": out_frames / TARGET_RATE,
        "peak_db": float(20.0 * np.log10(max(peak, 1e-8))),
    }
