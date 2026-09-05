"""FlashSR 后端测试：导入期替身、权重加载前置校验、单块推理契约。

运行：.venv-flashsr\\Scripts\\python.exe python/audio_ai_flashsr/test_backend.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

import backend  # noqa: E402

CHUNK = backend.FLASHSR_CHUNK_SAMPLES


class StubModel(torch.nn.Module):
    """最小替身模型：默认恒等，可指定输出以触发校验分支。

    构造签名模仿上游 `FlashSR(ldm, vocoder, vae)`：接收任意个权重路径，
    `output` 为关键字参数。
    """

    def __init__(self, *paths: Any, output: torch.Tensor | None = None) -> None:
        super().__init__()
        del paths
        self._output = output

    def forward(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return tensor if self._output is None else self._output


class ImportCompatibilityTests(unittest.TestCase):
    def test_stubs_are_installed_for_plotting_dependencies(self) -> None:
        """在干净子进程中断言：绘图依赖被替身取代，而非被真实加载。"""
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(MODULE_ROOT)!r});"
            "import backend;"
            "backend.ensure_inference_only_imports();"
            "names = ['matplotlib', 'matplotlib.pyplot', 'matplotlib.pylab',"
            "         'sklearn.manifold', 'librosa.display'];"
            "bad = [n for n in names"
            "       if n not in sys.modules or sys.modules[n].__spec__ is None"
            "       or sys.modules[n].__spec__.loader is not None];"
            "assert not bad, f'未生效的替身: {bad}';"
            "print('stubs ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stubs ok", result.stdout)

    def test_stub_spec_is_required_by_find_spec(self) -> None:
        """上游内联的 diffusers 用 find_spec 探测依赖，替身必须带 __spec__。"""
        backend.ensure_inference_only_imports()
        for name in ("matplotlib", "sklearn.manifold", "librosa.display"):
            with self.subTest(name=name):
                module = sys.modules.get(name)
                if module is None:
                    continue
                self.assertIsNotNone(getattr(module, "__spec__", None), name)


class LoadFlashSrTests(unittest.TestCase):
    def test_requires_exactly_three_weights(self) -> None:
        with self.assertRaisesRegex(backend.BackendError, "需要 3 个权重文件"):
            backend.load_flashsr((Path("a.pth"),), torch.device("cpu"))

    def test_reports_missing_weight_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = tuple(root / f"{name}.pth" for name in backend.WEIGHT_NAMES)
            (root / "student_ldm.pth").write_bytes(b"x")
            with self.assertRaisesRegex(backend.BackendError, "权重文件缺失"):
                backend.load_flashsr(paths, torch.device("cpu"))

    def test_loads_and_moves_model_to_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = tuple(root / f"{name}.pth" for name in backend.WEIGHT_NAMES)
            for path in paths:
                path.write_bytes(b"x")

            with patch.object(backend, "import_flashsr", return_value=StubModel):
                model = backend.load_flashsr(paths, torch.device("cpu"))

            self.assertIsInstance(model, torch.nn.Module)
            self.assertFalse(model.training)  # eval() 已调用

    def test_torch_load_fallback_restores_original_function(self) -> None:
        import torch as torch_module

        original = torch_module.load
        with backend.torch_load_fallback():
            self.assertIsNot(torch_module.load, original)
        self.assertIs(torch_module.load, original)


class InferFlashSrTests(unittest.TestCase):
    def _block(self, channels: int = 1) -> np.ndarray:
        return np.full((channels, CHUNK), 0.25, dtype=np.float32)

    def test_identity_round_trip(self) -> None:
        block = self._block()
        result = backend.infer_flashsr(StubModel(), block, torch.device("cpu"))
        self.assertEqual(result.shape, block.shape)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, block, atol=1e-6)

    def test_stereo_is_processed_in_one_forward(self) -> None:
        """上游把第 0 维当作 batch：立体声一次前向即可，无需逐声道。"""
        calls: list[torch.Tensor] = []

        class Recorder(StubModel):
            def forward(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                calls.append(tensor)
                return tensor

        block = self._block(channels=2)
        result = backend.infer_flashsr(Recorder(), block, torch.device("cpu"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0].shape), (2, CHUNK))
        self.assertEqual(result.shape, (2, CHUNK))

    def test_rejects_wrong_chunk_length(self) -> None:
        block = np.zeros((1, CHUNK - 1), dtype=np.float32)
        with self.assertRaisesRegex(backend.BackendError, "245760"):
            backend.infer_flashsr(StubModel(), block, torch.device("cpu"))

    def test_rejects_more_than_two_channels(self) -> None:
        block = np.zeros((3, CHUNK), dtype=np.float32)
        with self.assertRaisesRegex(backend.BackendError, "unsupported block shape"):
            backend.infer_flashsr(StubModel(), block, torch.device("cpu"))

    def test_rejects_shape_mismatch(self) -> None:
        block = self._block()
        output = torch.zeros((1, CHUNK + 8), dtype=torch.float32)
        with self.assertRaisesRegex(backend.BackendError, "输出形状"):
            backend.infer_flashsr(StubModel(output=output), block, torch.device("cpu"))

    def test_rejects_non_finite_output(self) -> None:
        block = self._block()
        output = torch.full((1, CHUNK), float("nan"), dtype=torch.float32)
        with self.assertRaisesRegex(backend.BackendError, "NaN"):
            backend.infer_flashsr(StubModel(output=output), block, torch.device("cpu"))

    def test_forwards_num_steps_and_lowpass_input(self) -> None:
        seen: dict[str, Any] = {}

        class Recorder(StubModel):
            def forward(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                seen.update(kwargs)
                return tensor

        backend.infer_flashsr(
            Recorder(),
            self._block(),
            torch.device("cpu"),
            num_steps=1,
            lowpass_input=False,
        )
        self.assertEqual(seen, {"num_steps": 1, "lowpass_input": False})


if __name__ == "__main__":
    unittest.main()
