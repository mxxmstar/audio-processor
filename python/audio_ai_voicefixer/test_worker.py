"""VoiceFixer Worker 的清单解析、参数校验与协议回归测试。

不需要权重文件，也不需要 torch —— 只依赖标准库。
运行：python python/audio_ai_voicefixer/test_worker.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

import worker  # noqa: E402

try:  # 重叠相加的用例只在带 numpy 的环境里运行（`.venv-voicefixer`）
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pipeline 顶层会导入 torch：无 torch 的环境跳过相关用例
    import pipeline as _PIPELINE
except Exception:  # noqa: BLE001  # pragma: no cover
    _PIPELINE = None  # type: ignore[assignment]

WEIGHT_NAMES = ("analysis", "vocoder")

#: 与清单 `files[].name` 对应的假权重内容
FILES = {
    "analysis": b"voicefixer-analysis-bytes",
    "vocoder": b"voicefixer-vocoder-bytes",
}


def _write_weights(model_dir: Path, contents: dict[str, bytes]) -> dict[str, str]:
    """写入权重文件，返回 `name -> sha256`。"""
    digests: dict[str, str] = {}
    target = model_dir / "cache" / "voicefixer-home" / ".cache" / "voicefixer"
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        path = target / name
        path.write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _manifest(model_dir: Path, digests: dict[str, str], **overrides: object) -> None:
    files = [
        {
            "name": name,
            "file": f"cache/voicefixer-home/.cache/voicefixer/{name}",
            "size_bytes": len(contents),
            "sha256": digests.get(name, "0" * 64),
            "source": f"https://example.invalid/{name}",
        }
        for name, contents in FILES.items()
    ]
    entry = {
        "id": "voicefixer",
        "backend": "voicefixer",
        "version": "test-version",
        "file": "cache/voicefixer-home",
        "files": files,
        "sample_rate": 44100,
    }
    entry.update(overrides)
    model_dir.joinpath("manifest.json").write_text(
        json.dumps({"manifest_version": 1, "models": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class ManifestTests(unittest.TestCase):
    def test_reads_multi_file_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)

            manifest = worker.read_manifest(model_dir)
            assert manifest is not None
            entry = manifest["voicefixer"]
            self.assertEqual(entry["backend"], "voicefixer")
            self.assertEqual(entry["version"], "test-version")
            self.assertEqual(
                [artifact.name for artifact in entry["artifacts"]], list(WEIGHT_NAMES)
            )

    def test_native_sample_rate_defaults_to_44100(self) -> None:
        """缺省 sample_rate 时按 44.1 kHz 处理（§3.5：不得默认 48k）。"""
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            model_dir.joinpath("manifest.json").write_text(
                model_dir.joinpath("manifest.json")
                .read_text(encoding="utf-8")
                .replace('"sample_rate": 44100', '"sample_rate": 0'),
                encoding="utf-8",
            )
            spec = worker.find_model("voicefixer", model_dir)
            self.assertEqual(spec.native_sample_rate, 44100)

    def test_single_file_entry_is_supported(self) -> None:
        """不带 `files` 的条目按单文件模型处理，与 audio_ai 的清单兼容。"""
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            model_dir.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "voicefixer",
                                "backend": "voicefixer",
                                "file": "weights.bin",
                                "sha256": "a" * 64,
                                "sample_rate": 44100,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = worker.read_manifest(model_dir)
            assert manifest is not None
            artifacts = manifest["voicefixer"]["artifacts"]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].file, "weights.bin")

    def test_rejects_empty_files_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            _manifest(model_dir, {}, files=[])
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.read_manifest(model_dir)
            self.assertEqual(raised.exception.code, "MODEL_MANIFEST_INVALID")

    def test_rejects_missing_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            _manifest(model_dir, {})
            text = model_dir.joinpath("manifest.json").read_text(encoding="utf-8")
            model_dir.joinpath("manifest.json").write_text(
                text.replace('"sha256"', '"_sha256"'), encoding="utf-8"
            )
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.read_manifest(model_dir)
            self.assertEqual(raised.exception.code, "MODEL_MANIFEST_INVALID")

    def test_skips_unsupported_backend_entries(self) -> None:
        """共享 manifest 含其它后端条目时跳过，避免引入 voicefixer 破坏既有功能。"""
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            voicefixer_entry = {
                "id": "voicefixer",
                "backend": "voicefixer",
                "version": "test-version",
                "file": "cache/voicefixer-home",
                "sample_rate": 44100,
                "files": [
                    {
                        "name": name,
                        "file": f"cache/voicefixer-home/.cache/voicefixer/{name}",
                        "size_bytes": len(contents),
                        "sha256": "0" * 64,
                        "source": f"https://example.invalid/{name}",
                    }
                    for name, contents in FILES.items()
                ],
            }
            foreign_entry = {
                "id": "flashsr",
                "backend": "flashsr",
                "version": "x",
                "file": "cache/flashsr/pytorch_model.bin",
                "size_bytes": 1,
                "sha256": "0" * 64,
                "source": "https://example.invalid/m.bin",
            }
            model_dir.joinpath("manifest.json").write_text(
                json.dumps({"models": [voicefixer_entry, foreign_entry]}, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = worker.read_manifest(model_dir)
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertIn("voicefixer", manifest)
            self.assertNotIn("flashsr", manifest)

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            _manifest(model_dir, {})
            text = model_dir.joinpath("manifest.json").read_text(encoding="utf-8")
            model_dir.joinpath("manifest.json").write_text(
                text.replace(
                    "cache/voicefixer-home/.cache/voicefixer/analysis", "../escaped.bin"
                ),
                encoding="utf-8",
            )
            models, errors = worker.available_models(model_dir)
            self.assertEqual(models, [])
            self.assertTrue(
                any("MODEL_MANIFEST_INVALID" in error for error in errors), errors
            )


class FindModelTests(unittest.TestCase):
    def test_find_model_verifies_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)

            spec = worker.find_model("voicefixer", model_dir)
            self.assertEqual(spec.model_id, "voicefixer")
            self.assertEqual(spec.version, "test-version")
            self.assertEqual([name for name, _ in spec.artifacts], list(WEIGHT_NAMES))

            weights = worker.ordered_weights(spec, WEIGHT_NAMES)
            self.assertEqual([path.name for path in weights], list(WEIGHT_NAMES))

    def test_find_model_detects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            digests["analysis"] = "0" * 64
            _manifest(model_dir, digests)
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("voicefixer", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_HASH_MISMATCH")

    def test_find_model_skips_hash_when_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            model_dir.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "voicefixer",
                                "backend": "voicefixer",
                                "name": "analysis",
                                "file": "cache/voicefixer-home/.cache/voicefixer/analysis",
                                "size_bytes": 0,
                                "sha256": worker.MODEL_HASH_DEFERRED,
                                "sample_rate": 44100,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            spec = worker.find_model("voicefixer", model_dir)
            self.assertEqual(spec.native_sample_rate, 44100)
            self.assertEqual(len(spec.artifacts), 1)

    def test_find_model_detects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            text = model_dir.joinpath("manifest.json").read_text(encoding="utf-8")
            text = text.replace(
                f'"size_bytes": {len(FILES["analysis"])}',
                f'"size_bytes": {len(FILES["analysis"]) + 1}',
            )
            model_dir.joinpath("manifest.json").write_text(text, encoding="utf-8")
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("voicefixer", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_SIZE_MISMATCH")

    def test_find_model_rejects_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("does-not-exist", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_NOT_FOUND")


class AvailableModelsTests(unittest.TestCase):
    def test_defers_hash_for_large_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            with patch.object(worker, "STARTUP_HASH_MAX_BYTES", 1):
                models, errors = worker.available_models(model_dir)
            self.assertEqual(models, ["voicefixer"])
            self.assertEqual(errors, ["voicefixer: MODEL_HASH_DEFERRED"])

    def test_reports_missing_weight_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, {})
            _manifest(model_dir, {"analysis": "0" * 64, "vocoder": "0" * 64})
            models, errors = worker.available_models(model_dir)
            self.assertEqual(models, [])
            self.assertEqual(errors, ["voicefixer: MODEL_NOT_FOUND"])

    def test_valid_weights_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            models, errors = worker.available_models(model_dir)
            self.assertEqual(models, ["voicefixer"])
            self.assertEqual(errors, [])


class ModeTests(unittest.TestCase):
    def test_default_mode_is_zero(self) -> None:
        self.assertEqual(worker.parse_mode({}), 0)
        self.assertEqual(worker.parse_mode({"mode": None}), 0)
        self.assertEqual(worker.parse_mode({"mode": ""}), 0)

    def test_accepts_official_modes(self) -> None:
        for mode in (0, 1, 2):
            self.assertEqual(worker.parse_mode({"mode": mode}), mode)
        # 前端可能以字符串下发
        self.assertEqual(worker.parse_mode({"mode": "2"}), 2)

    def test_rejects_unknown_mode(self) -> None:
        for raw in (3, -1, "all", "abc"):
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.parse_mode({"mode": raw})
            self.assertEqual(raised.exception.code, "UNSUPPORTED_MODE")


class LazyImportTests(unittest.TestCase):
    """R12：ready 握手之前绝不能导入 torch / numpy。

    Rust 侧的 READY_TIMEOUT 只有 3 秒，而 torch + voicefixer 导入实测约 7 秒。
    这里用独立子进程断言，避免被同一进程内其它测试污染。
    """

    def test_worker_module_has_no_heavy_imports(self) -> None:
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(MODULE_ROOT)!r});"
            "import worker;"
            "leaked = [m for m in ('torch', 'numpy', 'voicefixer') if m in sys.modules];"
            "assert not leaked, f'worker 顶层导入了重量级依赖: {leaked}';"
            "print('clean')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean", result.stdout)


class FakeWorkerProtocolTests(unittest.TestCase):
    """用假后端跑通 ready / progress / result 与取消路径（无需权重与 torch）。"""

    def _spawn(self, mode: str = "success") -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(MODULE_ROOT / "fake_worker.py"), "--mode", mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def test_success_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.wav"
            proc = self._spawn("success")
            try:
                assert proc.stdin is not None and proc.stdout is not None
                ready = json.loads(proc.stdout.readline())
                self.assertEqual(ready["type"], "ready")
                self.assertEqual(ready["models"], ["voicefixer"])

                proc.stdin.write(
                    json.dumps(
                        {
                            "command": "process",
                            "request_id": "req-1",
                            "input_path": "in.wav",
                            "output_path": str(output),
                            "model_id": "voicefixer",
                            "device": "cpu",
                            "mode": 0,
                        }
                    )
                    + "\n"
                )
                proc.stdin.flush()

                events = []
                while True:
                    line = proc.stdout.readline()
                    self.assertTrue(line, "worker 提前退出，未收到 result")
                    event = json.loads(line)
                    events.append(event)
                    if event["type"] == "result":
                        break
                progress = [e for e in events if e["type"] == "progress"]
                self.assertTrue(progress)
                self.assertEqual(progress[-1]["phase"], "encode")
                result = events[-1]
                # §3.5：假后端也必须报 44.1 kHz，否则掩盖 R2 类缺陷
                self.assertEqual(result["sample_rate"], 44100)
                self.assertEqual(result["model_id"], "voicefixer")
                self.assertTrue(output.is_file())
            finally:
                if proc.stdin and not proc.stdin.closed:
                    try:
                        proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, ValueError, OSError):
                        pass
                proc.wait(timeout=30)

    def test_cancel_stops_processing(self) -> None:
        proc = self._spawn("slow")
        try:
            assert proc.stdin is not None and proc.stdout is not None
            self.assertEqual(json.loads(proc.stdout.readline())["type"], "ready")
            proc.stdin.write(
                json.dumps({"command": "process", "request_id": "req-2", "output_path": ""})
                + "\n"
            )
            proc.stdin.flush()
            self.assertEqual(json.loads(proc.stdout.readline())["type"], "progress")
            # 等假后端跑几步再取消
            time.sleep(0.2)
            proc.stdin.write(json.dumps({"command": "cancel"}) + "\n")
            proc.stdin.flush()

            deadline = time.time() + 15
            cancelled = False
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break
                event = json.loads(line)
                if event["type"] == "error" and event["code"] == "CANCELLED":
                    cancelled = True
                    break
            self.assertTrue(cancelled, "取消后未收到 CANCELLED 事件")
        finally:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            proc.wait(timeout=30)


class _ListSink:
    """把写入的块收集起来，`OverlapAddWriter` 只依赖 `write()` 接口。"""

    def __init__(self) -> None:
        self.parts: list = []
        self.peak = 0.0

    def write(self, region) -> None:
        if region is None or region.size == 0:
            return
        self.parts.append(np.asarray(region, dtype=np.float32).copy())
        self.peak = max(self.peak, float(np.abs(region).max()))

    def close(self) -> None:
        pass

    def discard(self) -> None:
        pass


@unittest.skipUnless(_PIPELINE is not None, "pipeline 需要 numpy / torch 环境")
class OverlapAddTests(unittest.TestCase):
    """重叠相加的拼接逻辑（不需要 VoiceFixer 权重）。"""

    def test_total_length_matches_input(self) -> None:
        sink = _ListSink()
        writer = _PIPELINE.OverlapAddWriter(sink, 4)
        for index in range(3):
            writer.push(np.full(10, float(index + 1), dtype=np.float32), is_last=index == 2)
        writer.flush()
        # 块长 10、重叠 4 → 前进步长 6：输出 = 6 + 6 + 10 = 22（= 输入总长）
        self.assertEqual(sum(part.size for part in sink.parts), 22)

    def test_crossfade_is_linear(self) -> None:
        sink = _ListSink()
        writer = _PIPELINE.OverlapAddWriter(sink, 4)
        writer.push(np.zeros(10, dtype=np.float32), is_last=False)
        writer.push(np.ones(10, dtype=np.float32), is_last=True)
        writer.flush()
        joined = np.concatenate(sink.parts)
        np.testing.assert_allclose(joined[:6], np.zeros(6, dtype=np.float32))
        # 重叠区按 linspace(0,1,4) 线性混合，权重和为 1 → 无电平起伏
        np.testing.assert_allclose(joined[6:10], np.linspace(0.0, 1.0, 4, dtype=np.float32))
        np.testing.assert_allclose(joined[10:], np.ones(6, dtype=np.float32))

    def test_fade_zero_concatenates(self) -> None:
        sink = _ListSink()
        writer = _PIPELINE.OverlapAddWriter(sink, 0)
        writer.push(np.full(4, 1.0, dtype=np.float32), is_last=False)
        writer.push(np.full(4, 2.0, dtype=np.float32), is_last=True)
        writer.flush()
        np.testing.assert_allclose(
            np.concatenate(sink.parts), np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
