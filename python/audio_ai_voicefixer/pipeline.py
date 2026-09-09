"""VoiceFixer 推理流水线：探测、解码、分块修复（重叠相加）、编码。

本模块在收到 `process` 命令之后才被 `worker.py` 导入（风险 R12）：顶层
`import torch` + `import voicefixer` 实测约 7 秒，若放在启动时执行会超出
Rust 侧 3 秒的 ready 握手超时。清单解析留在 `worker.py`（纯标准库），这样
`from worker import find_model` 不会拉起 torch。

与 FlashSR / HiFi-GAN 的三个关键差异（见实施计划 §3.5 / §4.1）：

1. **输出固定 44.1 kHz**，不得套用 FlashSR 的「强制 48 kHz」校验（R2）；
2. **缓存目录必须重定向**：VoiceFixer 把权重路径硬编码为
   `~/.cache/voicefixer/...`，Windows 下 `expanduser("~")` 只认 `USERPROFILE`
   （设 `HOME` 无效），故在导入前改写环境变量（R1，§4.1 ②）；
3. **必须外层分块**：`restore_inmem()` 内部虽按 30 s 分段，但对外是黑盒、
   不可打断（R5）。实测切 10 s 块更快，但块间结果不一致（与整段相关性
   0.91 / 0.20），故块间做**重叠 + 线性交叉淡化**（R4，§4.1 ④）。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from protocol import (
    PROTOCOL_VERSION,
    SCRIPT_ROOT,
    WorkerFailure,
    emit_progress,
    log,
    resolve_binary,
)
from worker import (
    NATIVE_SAMPLE_RATE,
    ModelSpec,
    find_model,
    ordered_weights,
    parse_mode,
)

#: 清单里两个权重文件的 `name`（顺序即 `ordered_weights` 的取值顺序）。
MODEL_WEIGHT_NAME = ("analysis", "vocoder")

#: VoiceFixer 原生输出采样率。**不要**改成 48000（§3.5）。
TARGET_RATE = NATIVE_SAMPLE_RATE

# 编码时每个写块的目标采样点数（约 1 MB / 块，兼顾吞吐与内存占用）
ENCODE_BLOCK_SAMPLES = 262_144

#: 进度区间：与 audio_ai / FlashSR / HiFi-GAN 保持一致，前端无需区分后端
PERCENT_LOAD_MODEL = 5.0
PERCENT_PREPARE = 10.0
PERCENT_INFERENCE_START = 10.0
PERCENT_INFERENCE_SPAN = 80.0
PERCENT_WRITE_OUTPUT = 95.0

#: 默认块长（秒）。实测 10 s 块的总耗时低于整段（前向近似超线性），
#: 且块边界是唯一可检查取消的时机（§4.1 ④）。Rust 侧按后端下发 chunk_seconds。
DEFAULT_CHUNK_SECONDS = 10.0

#: 默认块间重叠（秒）。用于线性交叉淡化，掩盖块间结果差异造成的接缝。
DEFAULT_OVERLAP_SECONDS = 0.5

#: 运行时缓存：同一 (权重路径, 设备) 只加载一次模型。
_RUNNER_CACHE: dict[tuple[str, str], Any] = {}
_RUNNER_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# 缓存目录重定向（R1）
# --------------------------------------------------------------------------


def cache_home_for(model_dir: Path) -> Path:
    """权重缓存的「假 HOME」目录。

    VoiceFixer 固定把权重放到 `<home>/.cache/voicefixer/...`，因此只需把 home
    指到 `models/cache/voicefixer-home`，权重就整体落在 `models/cache/` 内
    （该目录已被 .gitignore 忽略），无需给上游代码打补丁。
    """
    return Path(model_dir) / "cache" / "voicefixer-home"


#: 上游硬编码的权重相对路径（`voicefixer/restorer/__init__.py` 与
#: `voicefixer/vocoder/__init__.py` 在**导入期**就会按这两个路径检查并
#: `urlretrieve` 自动下载，Zenodo 在本机返回 403，详见实施计划 §4.1 ②）。
ANALYSIS_WEIGHT_RELATIVE = ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt"
VOCODER_WEIGHT_RELATIVE = ".cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt"


def assert_weights_installed(home: Path) -> None:
    """导入前断言权重已就位，避免上游在导入期偷偷发起网络下载。

    `voicefixer` 的两个子包在模块导入时就会检查权重，缺失则直接用
    `urllib.request.urlretrieve` 拉 Zenodo（本机 403，且会阻塞数分钟、
    往 stdout 打印进度）。本项目要求权重由用户显式安装到 `models/cache/`，
    因此这里提前给出明确的 `MODEL_NOT_FOUND`。
    """
    missing = [
        relative
        for relative in (ANALYSIS_WEIGHT_RELATIVE, VOCODER_WEIGHT_RELATIVE)
        if not (home / relative).is_file()
    ]
    if missing:
        raise WorkerFailure(
            "MODEL_NOT_FOUND",
            "VoiceFixer 权重未安装，请先在「模型管理」中安装 voicefixer；缺失："
            + "、".join(missing),
        )


def configure_cache_home(model_dir: Path) -> Path:
    """把 `USERPROFILE` / `HOME` 指向模型缓存目录，返回该目录。

    必须在 `import voicefixer` **之前**调用：`voicefixer.vocoder.config.Config.ckpt`
    在模块导入期就对 `expanduser("~")` 求值。同时这两个变量必须在整个进程
    生命周期内保持 —— `VoiceFixer.__init__` 在构造时才求值分析模块路径，
    “导入完就恢复环境变量”会让分析模块落回真实用户目录。
    """
    home = cache_home_for(model_dir)
    home.mkdir(parents=True, exist_ok=True)
    value = str(home)
    # Windows 只看 USERPROFILE；POSIX 只看 HOME。两个都设，跨平台一致。
    os.environ["USERPROFILE"] = value
    os.environ["HOME"] = value
    return home


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
    """把输入音频解码为 `(channels, frames)` 的 float32（重采样到 44.1 kHz）。

    VoiceFixer 的分析模块按 44.1 kHz 训练，输入统一由 FFmpeg 重采样到该速率：
    低带宽样本（如 8 kHz）先被拉到 44.1 kHz 再修复，高带宽样本（48 kHz）会
    被降到 44.1 kHz —— 这是后端原生速率，不做二次重采样（§3.5）。
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
    # 削波到 [-1, 1]：极端样本会破坏 mel 提取，也让修复结果幅度失控
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


def encode_pcm_files(
    pcm_paths: list[Path],
    output_path: str,
    sample_rate: int,
    channels: int,
    gain: float = 1.0,
) -> None:
    """把 N 个单声道 f32le 暂存文件交错后编码为目标格式。

    逐块读取、逐块交错，内存中只保留一个块（长音频也不会把整段 PCM 驻留）。
    """
    block_frames = max(1, ENCODE_BLOCK_SAMPLES // max(1, channels))
    handles = [open(path, "rb") for path in pcm_paths]
    try:

        def blocks() -> Iterator[np.ndarray]:
            while True:
                columns = [
                    np.frombuffer(handle.read(block_frames * 4), dtype=np.float32)
                    for handle in handles
                ]
                frames = min(column.size for column in columns)
                if frames == 0:
                    return
                stacked = np.stack([column[:frames] for column in columns], axis=1)
                if gain != 1.0:
                    stacked = stacked * gain
                yield np.ascontiguousarray(stacked, dtype=np.float32)

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
    finally:
        for handle in handles:
            try:
                handle.close()
            except OSError:
                pass


class PcmChunkWriter:
    """把单声道 `(1, frames)` 音频块顺序追加到 f32le 暂存文件，并增量统计峰值。"""

    def __init__(self) -> None:
        self.peak = 0.0
        handle = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".f32le", prefix="audio-processor-pcm-", delete=False
        )
        self._handle = handle
        self.path = Path(handle.name)

    def write(self, region: np.ndarray | None) -> None:
        """写入一块定稿音频（1-D float32）；`None` / 空块表示无样本可写。"""
        if region is None or region.size == 0:
            return
        magnitude = float(np.abs(region).max())
        if magnitude > self.peak:
            self.peak = magnitude
        self._handle.write(np.ascontiguousarray(region, dtype=np.float32).tobytes())

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


class OverlapAddWriter:
    """把逐块修复结果按「重叠 + 线性交叉淡化」拼接后写出。

    §4.1 ④：分块能换来取消粒度与更低峰值内存，但相邻块的结果并不一致
    （与整段结果相关性仅 0.91 / 0.20），直接拼接会在边界产生可听的接缝。
    这里让每块尾部保留 `fade` 个样本，与下一块头部按 `a` 与 `1-a` 线性混合，
    权重和为 1（NOLA 条件成立），因此不会引入电平起伏。

    输出总长度 = 各块长度之和 − 重叠长度之和，与输入时长一致。
    """

    def __init__(self, sink: PcmChunkWriter, fade_frames: int) -> None:
        self._sink = sink
        self._fade = max(0, int(fade_frames))
        self._tail: np.ndarray | None = None

    def push(self, block: np.ndarray, is_last: bool) -> None:
        fade = self._fade
        size = int(block.size)
        if self._tail is None:
            keep = 0 if (is_last or fade == 0) else fade
            body = size - keep
            self._sink.write(block[:body])
            self._tail = block[body:].copy() if keep else None
            return
        count = min(fade, self._tail.size, size)
        ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
        self._sink.write(self._tail[:count] * (1.0 - ramp) + block[:count] * ramp)
        if is_last:
            self._sink.write(block[count:])
            self._tail = None
            return
        self._sink.write(block[count : size - fade])
        self._tail = block[size - fade :].copy()

    def flush(self) -> None:
        if self._tail is not None:
            self._sink.write(self._tail)
            self._tail = None


# --------------------------------------------------------------------------
# 设备与编排
# --------------------------------------------------------------------------


def warm_up_imports(model_dir: Path | None = None) -> None:
    """在主线程完成缓存重定向与上游导入，避免推理线程内死锁。

    两件事都必须在主线程做：

    1. **缓存重定向先于导入**（R1）：`Config.ckpt` 在导入期求值，
       顺序颠倒权重就会落到真实用户目录。
    2. **防死锁**（R12）：torch 会在导入时拉起线程池与动态库，若在推理线程
       内首次导入而主线程阻塞于 stdin 读取，存在死锁风险。

    上游导入期可能向 **stdout** 打印诊断信息（如 torch 的 FutureWarning 走
    stderr，`voicefixer` 的部分 warning 走 stdout 曾被观察到）。stdout 被
    JSONL 协议独占，混入非协议文本会让 Rust 侧解析失败，故统一重定向到
    stderr —— 与 `audio_ai_hifigan.pipeline.warm_up_imports` 的处理一致。

    这里只做导入，不加载权重（权重在 `enhance` 内按需加载）。
    """
    directory = Path(
        model_dir
        or os.environ.get("AUDIO_AI_MODEL_DIR")
        or (SCRIPT_ROOT / "models")
    )
    home = configure_cache_home(directory)
    # 必须在 import 之前：上游子包在导入期会自动下载缺失的权重
    assert_weights_installed(home)

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            import voicefixer  # noqa: F401
            import voicefixer.vocoder.config  # noqa: F401
    finally:
        noise = buffer.getvalue()
        if noise.strip():
            print(noise.rstrip(), file=sys.stderr, flush=True)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise WorkerFailure("CUDA_UNAVAILABLE", "CUDA is not available")
        return torch.device("cuda")
    if requested not in {"auto", "cpu"}:
        raise WorkerFailure("UNSUPPORTED_DEVICE", f"unsupported device: {requested}")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu")


def get_runner(spec: ModelSpec, device: torch.device) -> Any:
    """按 (权重路径, 设备) 缓存已加载的 VoiceFixer，避免重复加载。

    加载一次约 2.7 s（磁盘缓存命中）～81 s（冷启动读 466 MB ckpt），
    必须缓存。mode 在每次调用时传入，不参与缓存键。
    """
    weights = ordered_weights(spec, MODEL_WEIGHT_NAME)
    key = (str(weights[0]), str(weights[1]), str(device))
    with _RUNNER_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            from voicefixer import VoiceFixer

            # 构造时会读取 `~/.cache/voicefixer/analysis_module/checkpoints/vf.ckpt`，
            # 声码器权重在 Vocoder 构造时读 —— 两者都依赖已重定向的 home。
            runner = VoiceFixer()
            runner.to(device)
            _RUNNER_CACHE[key] = runner
        return runner


def restore_chunk(runner: Any, chunk: np.ndarray, mode: int, cuda: bool) -> np.ndarray:
    """修复一块单声道音频，返回 1-D float32（长度对齐输入）。

    `restore_inmem` 期望 float64 一维输入，返回形状 `(1, N)` 的数组。长度
    实测：mode 0 / 2 与输入精确一致；mode 1 因 `librosa.istft` 截断会短几百
    个样本（约 7 ms），此处补零 —— 缺口位于块尾，会被下一块的交叉淡化覆盖。
    """
    expected = int(chunk.size)
    with torch.no_grad():
        output = runner.restore_inmem(chunk.astype(np.float64), cuda=cuda, mode=mode)
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size > expected:
        values = values[:expected]
    elif values.size < expected:
        values = np.pad(values, (0, expected - values.size))
    if not np.isfinite(values).all():
        raise WorkerFailure("OUTPUT_INVALID", "restored audio contains NaN or Inf")
    return values


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _positive_float(request: dict[str, Any], key: str, default: float) -> float:
    raw = request.get(key)
    try:
        value = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        value = 0.0
    return value if value > 0 else default


def enhance(request: dict[str, Any], cancel_event: threading.Event) -> dict[str, Any]:
    """执行一次 VoiceFixer 语音修复，返回 `result` 事件载荷。"""
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
    mode = parse_mode(request)

    source_rate, channels, source_duration = probe_audio(input_path)
    if source_rate <= 0 or channels <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "invalid audio sample rate or channel count")

    # 分析/声码器面向单声道语音：超过两声道未定义，统一下混为 2。
    downmixed = False
    if channels > 2:
        log(f"输入为 {channels} 声道，超过 VoiceFixer 上限，下混为 2 声道")
        downmixed = True
        channels = 2

    model_dir = Path(os.environ.get("AUDIO_AI_MODEL_DIR", str(SCRIPT_ROOT / "models")))
    # 缓存重定向必须在任何 voicefixer 导入之后仍然成立（导入已完成也可能被
    # 其它代码路径触发），这里再兜一次底并确认权重就位。
    assert_weights_installed(configure_cache_home(model_dir))

    spec: ModelSpec = find_model(model_id, model_dir)
    device = choose_device(str(request.get("device", "auto")))

    # §3.5：VoiceFixer 原生 44.1 kHz，**不得**套用 FlashSR 的 48 kHz 校验（R2）。
    requested_sample_rate = request.get("output_sample_rate")
    if requested_sample_rate not in (None, "", TARGET_RATE):
        raise WorkerFailure(
            "UNSUPPORTED_SAMPLE_RATE",
            f"voicefixer produces {TARGET_RATE} Hz audio natively",
        )

    emit_progress(request_id, "load_model", PERCENT_LOAD_MODEL, message=f"loading {model_id}")
    runner = get_runner(spec, device)

    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")

    audio = decode_audio(input_path, TARGET_RATE, channels)
    frames = audio.shape[1]
    if frames <= 0:
        raise WorkerFailure("UNSUPPORTED_INPUT", "input audio contains no samples")

    if downmixed:
        emit_progress(
            request_id,
            "prepare_input",
            PERCENT_PREPARE,
            message=f"输入已下混为 {channels} 声道（VoiceFixer 最多支持 2 声道）",
        )
    else:
        emit_progress(request_id, "prepare_input", PERCENT_PREPARE, message="解码完成，开始修复")
    if cancel_event.is_set():
        raise WorkerFailure("CANCELLED", "processing cancelled")

    total_seconds = frames / TARGET_RATE
    chunk_seconds = _positive_float(request, "chunk_seconds", DEFAULT_CHUNK_SECONDS)
    overlap_seconds = _positive_float(request, "overlap_seconds", DEFAULT_OVERLAP_SECONDS)
    chunk_frames = max(1, int(chunk_seconds * TARGET_RATE))
    # 保证每块至少留得下 4 段淡入淡出区，避免 OverlapAddWriter 退化
    fade_frames = min(int(overlap_seconds * TARGET_RATE), max(0, chunk_frames // 4))
    starts = list(range(0, frames, chunk_frames - fade_frames)) if fade_frames else list(
        range(0, frames, chunk_frames)
    )
    if not starts:
        starts = [0]

    cuda = device.type == "cuda"
    sinks: list[PcmChunkWriter] = []
    try:
        for ch in range(channels):
            if cancel_event.is_set():
                raise WorkerFailure("CANCELLED", "processing cancelled")
            sink = PcmChunkWriter()
            sinks.append(sink)
            writer = OverlapAddWriter(sink, fade_frames)
            for index, start in enumerate(starts):
                if cancel_event.is_set():
                    raise WorkerFailure("CANCELLED", "processing cancelled")
                end = min(frames, start + chunk_frames)
                chunk = audio[ch, start:end].astype(np.float32)
                writer.push(restore_chunk(runner, chunk, mode, cuda), is_last=(end >= frames))
                done = (ch + (index + 1) / len(starts)) / channels
                emit_progress(
                    request_id,
                    "restore",
                    PERCENT_INFERENCE_START + PERCENT_INFERENCE_SPAN * done,
                    processed_seconds=total_seconds * done,
                    total_seconds=total_seconds or source_duration,
                    message=f"语音修复 mode {mode}（{index + 1}/{len(starts)} 块）",
                )
            writer.flush()
            sink.close()

        peak = max((sink.peak for sink in sinks), default=0.0)
        if not np.isfinite(peak) or peak <= 0.0:
            raise WorkerFailure("OUTPUT_INVALID", "restored audio is silent or invalid")
        gain = 1.0 / peak if peak > 1.0 else 1.0
        if peak > 1.0:
            peak = 1.0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        emit_progress(request_id, "encode", PERCENT_WRITE_OUTPUT, message="writing AI master")
        log(
            f"encode start: output={output_path} rate={TARGET_RATE} "
            f"channels={channels} gain={gain:.6f} peak={peak:.6f}"
        )
        written_frames = min(
            (sink.path.stat().st_size // 4 for sink in sinks), default=0
        )
        try:
            encode_pcm_files(
                [sink.path for sink in sinks],
                output_path,
                TARGET_RATE,
                channels,
                gain=gain,
            )
        finally:
            for sink in sinks:
                sink.discard()
            sinks = []
    finally:
        for sink in sinks:
            sink.discard()

    written = Path(output_path).stat().st_size if Path(output_path).is_file() else -1
    log(f"encode done: size={written}")
    if not Path(output_path).is_file() or written == 0:
        raise WorkerFailure("OUTPUT_INVALID", "AI output file is empty")

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
        "duration_seconds": written_frames / TARGET_RATE,
        "peak_db": float(20.0 * np.log10(max(peak, 1e-8))),
        "mode": mode,
    }
