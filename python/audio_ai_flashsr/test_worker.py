"""FlashSR Worker 的清单解析与协议原语测试。

不需要权重文件，也不需要 torch —— 只依赖标准库。
运行：python python/audio_ai_flashsr/test_worker.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT))

import worker  # noqa: E402

WEIGHT_NAMES = ("student_ldm", "sr_vocoder", "vae")


def _write_weights(model_dir: Path, contents: dict[str, bytes]) -> dict[str, str]:
    """写入权重文件，返回 `name -> sha256`。"""
    digests: dict[str, str] = {}
    target = model_dir / "cache" / "flashsr"
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        path = target / f"{name}.pth"
        path.write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _manifest(model_dir: Path, digests: dict[str, str], **overrides: object) -> None:
    files = [
        {
            "name": name,
            "file": f"cache/flashsr/{name}.pth",
            "size_bytes": len(contents),
            # 负面用例不传真实摘要，用占位值即可（校验目标是清单结构）
            "sha256": digests.get(name, "0" * 64),
            "source": f"https://example.invalid/{name}.pth",
        }
        for name, contents in FILES.items()
    ]
    entry = {
        "id": "flashsr",
        "backend": "flashsr",
        "version": "test-version",
        "file": "cache/flashsr",
        "files": files,
    }
    entry.update(overrides)
    model_dir.joinpath("manifest.json").write_text(
        json.dumps({"manifest_version": 1, "models": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


FILES = {
    "student_ldm": b"student-ldm-bytes",
    "sr_vocoder": b"sr-vocoder-bytes",
    "vae": b"vae-bytes",
}


class ManifestTests(unittest.TestCase):
    def test_reads_multi_file_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)

            manifest = worker.read_manifest(model_dir)
            assert manifest is not None
            entry = manifest["flashsr"]
            self.assertEqual(entry["backend"], "flashsr")
            self.assertEqual(entry["version"], "test-version")
            self.assertEqual(
                [artifact.name for artifact in entry["artifacts"]], list(WEIGHT_NAMES)
            )
            self.assertEqual(
                [artifact.file for artifact in entry["artifacts"]],
                [f"cache/flashsr/{name}.pth" for name in WEIGHT_NAMES],
            )

    def test_single_file_entry_is_supported(self) -> None:
        """不带 `files` 的条目按单文件模型处理，与 audio_ai 的清单兼容。"""
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            model_dir.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "flashsr",
                                "backend": "flashsr",
                                "file": "weights.pth",
                                "sha256": "a" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = worker.read_manifest(model_dir)
            assert manifest is not None
            artifacts = manifest["flashsr"]["artifacts"]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].file, "weights.pth")
            self.assertEqual(artifacts[0].size_bytes, 0)

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

    def test_rejects_non_positive_size_in_files_entry(self) -> None:
        """`size_bytes` 必须写在 files 条目内：有 files 时顶层字段被忽略。"""
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            model_dir.joinpath("manifest.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "flashsr",
                                "backend": "flashsr",
                                "file": "cache/flashsr",
                                "files": [
                                    {
                                        "name": "vae",
                                        "file": "cache/flashsr/vae.pth",
                                        "size_bytes": 0,
                                        "sha256": "0" * 64,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.read_manifest(model_dir)
            self.assertEqual(raised.exception.code, "MODEL_MANIFEST_INVALID")

    def test_rejects_unsupported_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            _manifest(model_dir, {}, backend="audiosr")
            with self.assertRaisesRegex(worker.WorkerFailure, "unsupported model backend"):
                worker.read_manifest(model_dir)

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_weights(model_dir, FILES)
            _manifest(model_dir, {})
            text = model_dir.joinpath("manifest.json").read_text(encoding="utf-8")
            model_dir.joinpath("manifest.json").write_text(
                text.replace("cache/flashsr/student_ldm.pth", "../escaped.pth"),
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

            spec = worker.find_model("flashsr", model_dir)
            self.assertEqual(spec.model_id, "flashsr")
            self.assertEqual(spec.version, "test-version")
            self.assertEqual([name for name, _ in spec.artifacts], list(WEIGHT_NAMES))

            weights = worker.ordered_weights(spec, WEIGHT_NAMES)
            self.assertEqual(
                [path.name for path in weights], [f"{name}.pth" for name in WEIGHT_NAMES]
            )

    def test_find_model_detects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            digests["vae"] = "0" * 64
            _manifest(model_dir, digests)
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("flashsr", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_HASH_MISMATCH")

    def test_find_model_detects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            text = model_dir.joinpath("manifest.json").read_text(encoding="utf-8")
            text = text.replace(
                f'"size_bytes": {len(FILES["vae"])}',
                f'"size_bytes": {len(FILES["vae"]) + 1}',
            )
            model_dir.joinpath("manifest.json").write_text(text, encoding="utf-8")
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("flashsr", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_SIZE_MISMATCH")

    def test_find_model_rejects_unknown_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("does-not-exist", model_dir)
            self.assertEqual(raised.exception.code, "MODEL_NOT_FOUND")

    def test_find_model_requires_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(worker.WorkerFailure) as raised:
                worker.find_model("flashsr", Path(directory))
            self.assertEqual(raised.exception.code, "MODEL_NOT_FOUND")

    def test_ordered_weights_rejects_missing_entry(self) -> None:
        spec = worker.ModelSpec(
            model_id="flashsr",
            version="v",
            backend="flashsr",
            model_name="",
            artifacts=(("student_ldm", Path("/tmp/student_ldm.pth")),),
        )
        with self.assertRaisesRegex(worker.WorkerFailure, "missing weight entries"):
            worker.ordered_weights(spec, WEIGHT_NAMES)

    def test_ordered_weights_ignores_declaration_order(self) -> None:
        """取权重按名称而非清单顺序，避免清单顺序写错导致权重错位。"""
        spec = worker.ModelSpec(
            model_id="flashsr",
            version="v",
            backend="flashsr",
            model_name="",
            artifacts=(
                ("vae", Path("/w/vae.pth")),
                ("student_ldm", Path("/w/student_ldm.pth")),
                ("sr_vocoder", Path("/w/sr_vocoder.pth")),
            ),
        )
        self.assertEqual(
            [path.name for path in worker.ordered_weights(spec, WEIGHT_NAMES)],
            ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"],
        )


class AvailableModelsTests(unittest.TestCase):
    def test_defers_hash_for_large_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            with patch.object(worker, "STARTUP_HASH_MAX_BYTES", 1):
                models, errors = worker.available_models(model_dir)
            self.assertEqual(models, ["flashsr"])
            self.assertEqual(errors, ["flashsr: MODEL_HASH_DEFERRED"])

    def test_reports_missing_weight_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, {k: v for k, v in FILES.items() if k != "vae"})
            digests["vae"] = "0" * 64
            _manifest(model_dir, digests)
            models, errors = worker.available_models(model_dir)
            self.assertEqual(models, [])
            self.assertEqual(errors, ["flashsr: MODEL_NOT_FOUND"])

    def test_valid_weights_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            digests = _write_weights(model_dir, FILES)
            _manifest(model_dir, digests)
            models, errors = worker.available_models(model_dir)
            self.assertEqual(models, ["flashsr"])
            self.assertEqual(errors, [])

    def test_missing_manifest_yields_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(worker.available_models(Path(directory)), ([], []))


class LazyImportTests(unittest.TestCase):
    """R12：ready 握手之前绝不能导入 torch / numpy。

    Rust 侧的 READY_TIMEOUT 只有 3 秒，而 torch 导入实测约 5.7 秒。
    这里用独立子进程断言，避免被同一进程内其它测试污染。
    """

    def _assert_clean_import(self) -> None:
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(MODULE_ROOT)!r});"
            "import worker;"
            "leaked = [m for m in ('torch', 'numpy') if m in sys.modules];"
            "assert not leaked, f'worker 顶层导入了重量级依赖: {leaked}';"
            "print('clean')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean", result.stdout)

    def test_worker_module_has_no_heavy_imports(self) -> None:
        self._assert_clean_import()


if __name__ == "__main__":
    unittest.main()
