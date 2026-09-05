"""FlashSR 流水线测试：定长分块、尾块补零、重叠相加、编码与峰值。

用桩替换 `backend` 的模型加载与推理，因此不需要 3.3 GB 权重。
探测与解码使用真实的 ffprobe/ffmpeg，保证与外部工具集成正确性。

运行：.venv-flashsr\\Scripts\\python.exe python/audio_ai_flashsr/test_pipeline.py
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

import backend  # noqa: E402
import pipeline  # noqa: E402
from protocol import WorkerFailure  # noqa: E402
from worker import ModelSpec  # noqa: E402

SAMPLE_RATE = 48_000
CHUNK = backend.FLASHSR_CHUNK_SAMPLES
WEIGHT_NAMES = backend.WEIGHT_NAMES


def write_wav(path: Path, frames: int, channels: int = 1, amplitude: float = 0.3) -> None:
    """写入一段正弦波 PCM，便于在不依赖 ffmpeg 生成的前提下做数值断言。"""
    tones = np.sin(2 * math.pi * 440.0 * np.arange(frames, dtype=np.float32) / SAMPLE_RATE)
    samples = tones[:, None] * amplitude * np.ones((1, channels), dtype=np.float32)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def fake_spec(channels: int = 1, model_id: str = "flashsr") -> ModelSpec:
    artifacts = tuple(
        (name, Path(tempfile.gettempdir()) / f"{name}.pth") for name in WEIGHT_NAMES
    )
    return ModelSpec(
        model_id=model_id,
        version="test-version",
        backend="flashsr",
        model_name="",
        artifacts=artifacts,
    )


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.blocks: list[np.ndarray] = []
        self.gain = 2.0

    def tearDown(self) -> None:
        self.stop_patches()
        self.directory.cleanup()

    def _stub_inference(self) -> None:
        """记录收到的块，并返回放大后的结果，便于验证数值链路。"""

        def infer(model: Any, block: np.ndarray, device: Any, **kwargs: Any) -> np.ndarray:
            self.blocks.append(block.copy())
            return (block * self.gain).astype(np.float32)

        self._patchers = [
            patch.object(backend, "cached_flashsr", return_value="stub-model"),
            patch.object(backend, "infer_flashsr", side_effect=infer),
            patch.object(pipeline, "find_model", return_value=fake_spec()),
        ]
        for patcher in self._patchers:
            patcher.start()

    def stop_patches(self) -> None:
        for patcher in getattr(self, "_patchers", []):
            patcher.stop()

    def _request(self, input_path: Path, output_path: Path, **overrides: Any) -> dict[str, Any]:
        request = {
            "request_id": "req-test",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model_id": "flashsr",
            "device": "cpu",
            "overlap_seconds": 0.0,
            "output_sample_rate": SAMPLE_RATE,
        }
        request.update(overrides)
        return request

    # ---------------------------------------------------------------- 正常链路

    def test_enhance_feeds_fixed_length_blocks_and_pads_tail(self) -> None:
        """所有块都必须是定长 245760；尾块补零而不是送入更短的张量。"""
        self._stub_inference()
        try:
            frames = CHUNK * 2 + 1000
            source = self.root / "input.wav"
            write_wav(source, frames)
            output = self.root / "enhanced.flac"

            result = pipeline.enhance(
                self._request(source, output), threading.Event()
            )
        finally:
            self.stop_patches()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["channels"], 1)
        self.assertAlmostEqual(result["duration_seconds"], frames / SAMPLE_RATE, places=6)

        # step = 245760（overlap=0），因此 492520 帧被切成 3 块
        self.assertEqual(len(self.blocks), 3)
        for block in self.blocks:
            self.assertEqual(block.shape, (1, CHUNK))
        # 尾块：前 1000 个样本有数据，其余补零
        tail = self.blocks[-1]
        self.assertGreater(np.abs(tail[:, :1000]).max(), 0.0)
        self.assertEqual(np.abs(tail[:, 1000:]).max(), 0.0)

        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 0)

    def test_enhance_reports_peak_after_amplification(self) -> None:
        """峰值必须在落盘时统计：输入 0.3、增益 2.0 → 峰值 0.6。"""
        self.gain = 2.0
        self._stub_inference()
        try:
            source = self.root / "input.wav"
            write_wav(source, CHUNK, amplitude=0.3)
            output = self.root / "enhanced.flac"
            result = pipeline.enhance(self._request(source, output), threading.Event())
        finally:
            self.stop_patches()

        expected_db = 20.0 * math.log10(0.6)
        self.assertAlmostEqual(result["peak_db"], expected_db, places=2)

    def test_enhance_applies_overlap_crossfade(self) -> None:
        """有重叠时，重叠区的淡入淡出必须让拼接结果保持连续。"""
        self._stub_inference()
        try:
            frames = CHUNK * 2
            source = self.root / "input.wav"
            write_wav(source, frames)
            output = self.root / "enhanced.flac"
            result = pipeline.enhance(
                self._request(source, output, overlap_seconds=0.5), threading.Event()
            )
        finally:
            self.stop_patches()

        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["duration_seconds"], frames / SAMPLE_RATE, places=6)
        # step = 245760 - 24000 = 221760 → 起点 0 / 221760 / 443520
        self.assertEqual(len(self.blocks), 3)

    # ---------------------------------------------------------------- 声道处理

    def test_enhance_downmixes_more_than_two_channels(self) -> None:
        """上游按 [B, T] 处理，超过两声道未定义，必须下混为 2。"""
        self._stub_inference()
        try:
            source = self.root / "input.wav"
            write_wav(source, CHUNK, channels=3)
            output = self.root / "enhanced.flac"
            result = pipeline.enhance(self._request(source, output), threading.Event())
        finally:
            self.stop_patches()

        self.assertEqual(result["channels"], 2)
        self.assertTrue(all(block.shape == (2, CHUNK) for block in self.blocks))

    # ---------------------------------------------------------------- 参数校验

    def test_enhance_rejects_non_48k_output(self) -> None:
        source = self.root / "input.wav"
        write_wav(source, CHUNK)
        with self.assertRaises(WorkerFailure) as raised:
            pipeline.enhance(
                self._request(source, self.root / "out.flac", output_sample_rate=44100),
                threading.Event(),
            )
        self.assertEqual(raised.exception.code, "UNSUPPORTED_SAMPLE_RATE")

    def test_enhance_rejects_missing_input(self) -> None:
        with self.assertRaises(WorkerFailure) as raised:
            pipeline.enhance(
                self._request(self.root / "nope.wav", self.root / "out.flac"),
                threading.Event(),
            )
        self.assertEqual(raised.exception.code, "UNSUPPORTED_INPUT")

    def test_enhance_rejects_output_equal_to_input(self) -> None:
        source = self.root / "input.wav"
        write_wav(source, CHUNK)
        with self.assertRaises(WorkerFailure) as raised:
            pipeline.enhance(self._request(source, source), threading.Event())
        self.assertEqual(raised.exception.code, "INVALID_OUTPUT")

    def test_enhance_rejects_num_steps_below_one(self) -> None:
        self._stub_inference()
        try:
            source = self.root / "input.wav"
            write_wav(source, CHUNK)
            with self.assertRaises(WorkerFailure) as raised:
                pipeline.enhance(
                    self._request(source, self.root / "out.flac", num_steps=0),
                    threading.Event(),
                )
        finally:
            self.stop_patches()
        self.assertEqual(raised.exception.code, "INVALID_REQUEST")

    def test_enhance_honours_cancellation(self) -> None:
        self._stub_inference()
        try:
            source = self.root / "input.wav"
            write_wav(source, CHUNK * 3)
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(WorkerFailure) as raised:
                pipeline.enhance(
                    self._request(source, self.root / "out.flac"), cancel
                )
        finally:
            self.stop_patches()
        self.assertEqual(raised.exception.code, "CANCELLED")
        self.assertFalse((self.root / "out.flac").exists())

    def test_enhance_removes_temporary_pcm_on_failure(self) -> None:
        """失败路径不能把数百 MB 的暂存 PCM 留在系统临时目录。"""
        created: list[Path] = []
        original_init = pipeline.PcmChunkWriter.__init__

        def tracking_init(self: pipeline.PcmChunkWriter, channels: int) -> None:
            original_init(self, channels)
            created.append(self.path)

        self._stub_inference()
        try:
            source = self.root / "input.wav"
            write_wav(source, CHUNK * 2)
            with patch.object(pipeline.PcmChunkWriter, "__init__", tracking_init):
                with patch.object(backend, "infer_flashsr", side_effect=RuntimeError("boom")):
                    with self.assertRaises(RuntimeError):
                        pipeline.enhance(
                            self._request(source, self.root / "out.flac"), threading.Event()
                        )
        finally:
            self.stop_patches()

        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())


class ChunkAssemblerTests(unittest.TestCase):
    def test_overlapping_regions_average_by_weight(self) -> None:
        assembler = pipeline.ChunkAssembler(channels=1)
        blend = np.array([0.5, 0.5], dtype=np.float32)
        assembler.add(0, 2, np.array([[1.0, 1.0]], dtype=np.float32), blend)
        assembler.add(1, 3, np.array([[3.0, 3.0]], dtype=np.float32), blend)
        region = assembler.remaining()
        assert region is not None
        # 位置 0 只有第一块贡献：1.0*0.5/0.5 = 1.0
        # 位置 1 两块各占 0.5：(1*0.5 + 3*0.5)/1.0 = 2.0
        np.testing.assert_allclose(region, [[1.0, 2.0, 3.0]], atol=1e-6)

    def test_take_final_returns_none_when_nothing_is_final(self) -> None:
        assembler = pipeline.ChunkAssembler(channels=1)
        self.assertIsNone(assembler.take_final(0))
        self.assertIsNone(assembler.remaining())


if __name__ == "__main__":
    unittest.main()
