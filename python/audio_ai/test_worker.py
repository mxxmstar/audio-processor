"""Focused tests for the AudioSR worker adapter without model weights."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from audio_ai import worker


class AudioSrAdapterTests(unittest.TestCase):
    def test_manifest_requires_audiosr_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            model_dir.joinpath("manifest.json").write_text(
                """{
                    "models": [{
                        "id": "audiosr-basic",
                        "backend": "audiosr",
                        "file": "checkpoint.bin",
                        "sha256": "abc"
                    }]
                }""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(worker.WorkerFailure, "model_name"):
                worker.read_manifest(model_dir)

    def test_find_model_rejects_partial_audiosr_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoint = model_dir / "checkpoint.bin"
            checkpoint.write_bytes(b"too short")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model_dir.joinpath("manifest.json").write_text(
                f"""{{
                    "models": [{{
                        "id": "audiosr-basic",
                        "backend": "audiosr",
                        "model_name": "basic",
                        "file": "{checkpoint.name}",
                        "size_bytes": 999,
                        "sha256": "{digest}"
                    }}]
                }}""",
                encoding="utf-8",
            )
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("audiosr-basic", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_SIZE_MISMATCH")

    def test_available_models_defers_large_checkpoint_hash_until_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkpoint = model_dir / "checkpoint.bin"
            checkpoint.write_bytes(b"complete checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model_dir.joinpath("manifest.json").write_text(
                f"""{{
                    "models": [{{
                        "id": "audiosr-basic",
                        "backend": "audiosr",
                        "model_name": "basic",
                        "file": "{checkpoint.name}",
                        "sha256": "{digest}"
                    }}]
                }}""",
                encoding="utf-8",
            )
            with patch.object(worker, "STARTUP_HASH_MAX_BYTES", 1):
                models, errors = worker.available_models(model_dir)

            self.assertEqual(models, ["audiosr-basic"])
            self.assertEqual(errors, ["audiosr-basic: MODEL_HASH_DEFERRED"])

    def test_audiosr_plotting_compatibility_avoids_optional_dependency(self) -> None:
        with patch.dict(sys.modules, {"matplotlib": None}, clear=False):
            sys.modules.pop("matplotlib.pyplot", None)
            worker.ensure_audiosr_plotting_compatibility()
            self.assertTrue(callable(sys.modules["matplotlib"].use))
            self.assertIn("matplotlib.pyplot", sys.modules)

    def test_load_audiosr_uses_local_checkpoint_and_restores_downloader(self) -> None:
        checkpoint = Path("C:/models/audiosr-basic.bin")
        pipeline = types.ModuleType("audiosr.pipeline")
        calls: list[object] = []

        def remote_download(model_name: str) -> str:
            raise AssertionError(f"unexpected network download: {model_name}")

        def build_model(*, device: torch.device, model_name: str) -> object:
            calls.extend([device, model_name, pipeline.download_checkpoint(model_name)])
            print("AudioSR loading log")
            return "local-model"

        def super_resolution(*args: object, **kwargs: object) -> object:
            raise AssertionError("not used by this test")

        pipeline.download_checkpoint = remote_download
        pipeline.build_model = build_model
        pipeline.super_resolution = super_resolution
        package = types.ModuleType("audiosr")
        package.pipeline = pipeline
        with patch.dict(
            sys.modules, {"audiosr": package, "audiosr.pipeline": pipeline}
        ):
            model, inference = worker.load_audiosr(
                checkpoint, "basic", torch.device("cpu")
            )

        self.assertEqual(model, "local-model")
        self.assertIs(inference, super_resolution)
        self.assertEqual(calls, [torch.device("cpu"), "basic", str(checkpoint)])
        self.assertIs(pipeline.download_checkpoint, remote_download)

    def test_prepare_model_for_request_preloads_native_backends(self) -> None:
        model_dir = Path("C:/models")
        model_path = model_dir / "checkpoint.bin"
        runtime_path = model_dir / "runtime"
        device = torch.device("cpu")
        request = {"model_id": "model-id", "device": "cpu"}

        with patch.object(worker, "choose_device", return_value=device), patch.object(
            worker,
            "find_model",
            return_value=(model_path, "version", "audiosr", runtime_path, "basic"),
        ), patch.object(worker, "cached_audiosr") as cached_audiosr:
            worker.prepare_model_for_request(request, model_dir)

        cached_audiosr.assert_called_once_with(model_path, "basic", device)

        with patch.object(worker, "choose_device", return_value=device), patch.object(
            worker,
            "find_model",
            return_value=(model_path, "version", "deepfilternet", runtime_path, ""),
        ), patch.object(worker, "cached_deepfilternet") as cached_deepfilternet:
            worker.prepare_model_for_request(request, model_dir)

        cached_deepfilternet.assert_called_once_with(runtime_path, device)

    def test_audiosr_offline_patch_avoids_hub_and_restores_transformers(self) -> None:
        from transformers import RobertaConfig, RobertaTokenizer

        original_config_factory = RobertaConfig.__dict__.get("from_pretrained")
        original_tokenizer_factory = RobertaTokenizer.__dict__.get("from_pretrained")
        patches = worker.patch_audiosr_offline_dependencies()
        try:
            config = RobertaConfig.from_pretrained("missing-roberta")
            tokenizer = RobertaTokenizer.from_pretrained("missing-roberta")
            self.assertEqual(config.vocab_size, 50265)
            self.assertEqual(config.max_position_embeddings, 514)
            self.assertEqual(config.type_vocab_size, 1)
            encoded = tokenizer([""], max_length=4)
            self.assertEqual(tuple(encoded["input_ids"].shape), (1, 4))
            self.assertEqual(encoded["input_ids"].tolist(), [[0, 2, 1, 1]])
        finally:
            worker.restore_audiosr_offline_dependencies(patches)

        self.assertIs(RobertaConfig.__dict__.get("from_pretrained"), original_config_factory)
        self.assertIs(
            RobertaTokenizer.__dict__.get("from_pretrained"), original_tokenizer_factory
        )

    def test_infer_audiosr_trims_internal_padding(self) -> None:
        chunk = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        written: list[tuple[np.ndarray, str, int, int]] = []

        def fake_encode(
            audio: np.ndarray, output_path: str, sample_rate: int, channels: int
        ) -> None:
            written.append((audio.copy(), output_path, sample_rate, channels))
            Path(output_path).write_bytes(b"wave")

        def fake_super_resolution(*args: object, **kwargs: object) -> np.ndarray:
            return np.array([[[0.5, 0.4, 0.3, 0.2]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, "encode_audio", fake_encode):
                result = worker.infer_audiosr(
                    "model",
                    fake_super_resolution,
                    chunk,
                    Path(directory),
                    0,
                    1,
                )

        np.testing.assert_allclose(result, [[0.5, 0.4, 0.3]])
        self.assertEqual(len(written), 1)
        np.testing.assert_allclose(written[0][0], chunk)
        self.assertEqual(written[0][2:], (48_000, 1))

    def test_infer_audiosr_rejects_short_output(self) -> None:
        chunk = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        def fake_encode(
            audio: np.ndarray, output_path: str, sample_rate: int, channels: int
        ) -> None:
            Path(output_path).write_bytes(b"wave")

        def fake_super_resolution(*args: object, **kwargs: object) -> np.ndarray:
            return np.array([[[0.5, 0.4]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, "encode_audio", fake_encode):
                with self.assertRaisesRegex(worker.WorkerFailure, "shorter"):
                    worker.infer_audiosr(
                        "model",
                        fake_super_resolution,
                        chunk,
                        Path(directory),
                        0,
                        0,
                    )

    def test_infer_audiosr_reads_temporary_wav_without_torchcodec(self) -> None:
        import torchaudio

        chunk = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
        original_load = torchaudio.load
        loaded: list[tuple[torch.Tensor, int]] = []

        def fake_encode(
            audio: np.ndarray, output_path: str, sample_rate: int, channels: int
        ) -> None:
            pcm = np.clip(audio.T * 32767.0, -32768, 32767).astype("<i2")
            with wave.open(output_path, "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm.tobytes())

        def fake_super_resolution(*args: object, **kwargs: object) -> np.ndarray:
            waveform, sample_rate = torchaudio.load(args[1])
            loaded.append((waveform, sample_rate))
            return np.array([[[0.5, 0.4, 0.3]]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, "encode_audio", fake_encode):
                result = worker.infer_audiosr(
                    "model",
                    fake_super_resolution,
                    chunk,
                    Path(directory),
                    0,
                    0,
                )

        self.assertIs(torchaudio.load, original_load)
        self.assertEqual(loaded[0][1], 48_000)
        np.testing.assert_allclose(loaded[0][0].numpy(), chunk, atol=1e-4)
        np.testing.assert_allclose(result, [[0.5, 0.4, 0.3]])


class ChunkedWriteTests(unittest.TestCase):
    """分块写盘路径：定稿落盘、增量峰值统计、暂存文件清理。

    这批测试覆盖 `ChunkAssembler` + `PcmChunkWriter` + `encode_pcm_file`
    组成的整条写盘链路。AudioSR 端到端跑一次要几分钟，无法作为回归手段，
    因此用可真实加载的 TorchScript 恒等模型把 `enhance()` 跑通。
    """

    def test_writer_appends_interleaved_frames_and_tracks_peak(self) -> None:
        writer = worker.PcmChunkWriter(channels=2)
        try:
            first = np.array([[0.5, -0.25], [0.0, 0.75]], dtype=np.float32)
            second = np.array([[1.5, 0.0], [-0.5, 0.0]], dtype=np.float32)
            writer.write(None)
            writer.write(first)
            writer.write(np.zeros((2, 0), dtype=np.float32))
            writer.write(second)
            writer.close()

            self.assertAlmostEqual(writer.peak, 1.5, places=6)
            raw = writer.path.read_bytes()
            expected = np.ascontiguousarray(
                np.concatenate([first, second], axis=1).T, dtype=np.float32
            ).tobytes()
            self.assertEqual(raw, expected)
        finally:
            writer.discard()
        self.assertFalse(writer.path.exists())

    def test_writer_rejects_nan_or_inf_output(self) -> None:
        """回归：归一化结果含 NaN/Inf 必须抛 OUTPUT_INVALID，不能静默写盘。"""
        writer = worker.PcmChunkWriter(channels=1)
        try:
            with self.assertRaises(worker.WorkerFailure) as context:
                writer.write(np.array([[0.5, np.nan, 0.3]], dtype=np.float32))
            self.assertEqual(context.exception.code, "OUTPUT_INVALID")
            with self.assertRaises(worker.WorkerFailure) as context:
                writer.write(np.array([[0.5, np.inf]], dtype=np.float32))
            self.assertEqual(context.exception.code, "OUTPUT_INVALID")
        finally:
            writer.discard()

    def test_writer_discard_is_idempotent_after_close(self) -> None:
        writer = worker.PcmChunkWriter(channels=1)
        writer.write(np.array([[0.1, 0.2]], dtype=np.float32))
        writer.close()
        writer.discard()
        writer.discard()
        self.assertFalse(writer.path.exists())

    def _identity_checkpoint(self, directory: Path) -> Path:
        """保存一个恒等 TorchScript 模型，供 torch.jit.load 真实加载。"""

        class Identity(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        path = directory / "identity.ts"
        torch.jit.script(Identity()).save(str(path))
        return path

    def test_enhance_round_trips_audio_and_reports_peak(self) -> None:
        sample_rate = 48_000
        channels = 1
        duration_seconds = 0.5
        frames = int(sample_rate * duration_seconds)
        amplitude = 0.5
        audio = np.full((channels, frames), amplitude, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._identity_checkpoint(root)
            output_path = root / "enhanced.flac"
            request = {
                "request_id": "test-chunked",
                "input_path": str(root / "input.wav"),
                "output_path": str(output_path),
                "model_id": "identity",
                "device": "cpu",
                "chunk_seconds": 0.2,
                "overlap_seconds": 0.05,
                "output_sample_rate": sample_rate,
            }
            Path(request["input_path"]).write_bytes(b"stub")

            with patch.object(
                worker, "probe_audio", return_value=(sample_rate, channels, duration_seconds)
            ), patch.object(worker, "decode_audio", return_value=audio), patch.object(
                worker,
                "find_model",
                return_value=(checkpoint, "test-version", "torchscript", checkpoint, ""),
            ):
                result = worker.enhance(request, threading.Event())
                # 输出位于临时目录内，必须在目录销毁前完成文件校验
                output = Path(result["output_path"])
                output_exists = output.is_file()
                output_size = output.stat().st_size if output_exists else 0

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["sample_rate"], sample_rate)
        self.assertEqual(result["channels"], channels)
        self.assertAlmostEqual(result["duration_seconds"], duration_seconds, places=6)
        # 恒等模型不改变信号，峰值即输入幅度：20*log10(0.5) ≈ -6.02 dB。
        # 修复前 peak 恒为 0.0，这里会得到 -160 dB。
        self.assertAlmostEqual(result["peak_db"], -6.0206, places=3)
        self.assertTrue(output_exists)
        self.assertGreater(output_size, 0)

    def test_enhance_accepts_zero_overlap(self) -> None:
        """overlapSeconds == 0 是 Rust 侧允许的输入，必须能跑通。

        回归用例：`blend[-min(overlap, valid):]` 在 overlap 为 0 时等于
        `blend[0:]`（整个数组），与长度为 0 的 linspace 相乘会抛
        broadcast 错误，导致任务以 INFERENCE_FAILED 失败。
        """
        sample_rate = 48_000
        frames = 24_000
        audio = np.full((1, frames), 0.4, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._identity_checkpoint(root)
            output_path = root / "enhanced.flac"
            request = {
                "request_id": "test-zero-overlap",
                "input_path": str(root / "input.wav"),
                "output_path": str(output_path),
                "model_id": "identity",
                "device": "cpu",
                "chunk_seconds": 0.25,
                "overlap_seconds": 0.0,
                "output_sample_rate": sample_rate,
            }
            Path(request["input_path"]).write_bytes(b"stub")

            with patch.object(
                worker, "probe_audio", return_value=(sample_rate, 1, frames / sample_rate)
            ), patch.object(worker, "decode_audio", return_value=audio), patch.object(
                worker,
                "find_model",
                return_value=(checkpoint, "v", "torchscript", checkpoint, ""),
            ):
                result = worker.enhance(request, threading.Event())

        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["duration_seconds"], frames / sample_rate, places=6)

    def test_enhance_removes_temporary_pcm_on_failure(self) -> None:
        """失败路径同样要清掉暂存 PCM，否则会在系统临时目录堆积。"""
        sample_rate = 48_000
        audio = np.full((1, 4800), 0.5, dtype=np.float32)
        created: list[Path] = []

        real_init = worker.PcmChunkWriter.__init__

        def tracking_init(self: worker.PcmChunkWriter, channels: int) -> None:
            real_init(self, channels)
            created.append(self.path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._identity_checkpoint(root)
            request = {
                "request_id": "test-failure",
                "input_path": str(root / "input.wav"),
                "output_path": str(root / "out" / "enhanced.flac"),
                "model_id": "identity",
                "device": "cpu",
                "chunk_seconds": 0.2,
                "overlap_seconds": 0.05,
                "output_sample_rate": sample_rate,
            }
            Path(request["input_path"]).write_bytes(b"stub")

            def fail_during_inference(*args: object, **kwargs: object) -> None:
                # 只在推理阶段失败：load_model 阶段的进度上报早于 sink 创建，
                # 在那里抛错就测不到"暂存文件是否泄漏"。
                if kwargs.get("phase") == "inference" or (
                    len(args) > 1 and args[1] == "inference"
                ):
                    raise RuntimeError("boom")

            with patch.object(
                worker, "probe_audio", return_value=(sample_rate, 1, 0.1)
            ), patch.object(worker, "decode_audio", return_value=audio), patch.object(
                worker,
                "find_model",
                return_value=(checkpoint, "v", "torchscript", checkpoint, ""),
            ), patch.object(
                worker, "emit_progress", side_effect=fail_during_inference
            ), patch.object(
                worker.PcmChunkWriter, "__init__", tracking_init
            ):
                with self.assertRaises(RuntimeError):
                    worker.enhance(request, threading.Event())

        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())


if __name__ == "__main__":
    unittest.main()
