"""Focused tests for the AudioSR worker adapter without model weights."""

from __future__ import annotations

import hashlib
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
