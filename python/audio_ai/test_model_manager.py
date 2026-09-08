"""Tests for explicit model installation without network access."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_ai.model_manager import (
    ModelInstallError,
    install_model,
    read_model_entry,
    verify_file,
)


def _write_manifest(model_dir: Path, models: list[dict]) -> None:
    model_dir.joinpath("manifest.json").write_text(
        json.dumps({"models": models}, ensure_ascii=False), encoding="utf-8"
    )


class ModelManagerTests(unittest.TestCase):
    def test_read_model_entry_rejects_non_https_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            model_dir.joinpath("manifest.json").write_text(
                '{"models":[{"id":"test","file":"cache/model.bin",'
                '"source":"http://example.invalid/model.bin",'
                '"size_bytes":1,"sha256":"'
                + ("0" * 64)
                + '"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelInstallError, "HTTPS"):
                read_model_entry(model_dir, "test")

    def test_read_model_entry_parses_files_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_manifest(
                model_dir,
                [
                    {
                        "id": "flashsr",
                        "backend": "flashsr",
                        "file": "cache/flashsr",
                        "files": [
                            {
                                "name": name,
                                "file": f"cache/flashsr/{name}.pth",
                                "size_bytes": length,
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "source": f"https://example.invalid/{name}.pth",
                            }
                            for name, length, payload in (
                                ("student_ldm", 10, b"student"),
                                ("sr_vocoder", 20, b"vocoder"),
                                ("vae", 5, b"vae"),
                            )
                        ],
                    }
                ],
            )
            entry = read_model_entry(model_dir, "flashsr")
            artifacts = entry["artifacts"]
            self.assertEqual(
                [artifact["name"] for artifact in artifacts],
                ["student_ldm", "sr_vocoder", "vae"],
            )
            self.assertEqual(artifacts[0]["size_bytes"], 10)
            self.assertEqual(artifacts[2]["size_bytes"], 5)

    def test_read_model_entry_rejects_empty_files_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_manifest(
                model_dir,
                [{"id": "flashsr", "backend": "flashsr", "file": "cache/flashsr", "files": []}],
            )
            with self.assertRaisesRegex(ModelInstallError, "files list"):
                read_model_entry(model_dir, "flashsr")

    def test_read_model_entry_rejects_missing_artifact_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_manifest(
                model_dir,
                [
                    {
                        "id": "flashsr",
                        "backend": "flashsr",
                        "file": "cache/flashsr",
                        "files": [
                            {"name": "vae", "file": "cache/flashsr/vae.pth", "size_bytes": 1}
                        ],
                    }
                ],
            )
            with self.assertRaisesRegex(ModelInstallError, "field is missing"):
                read_model_entry(model_dir, "flashsr")

    def test_install_model_installs_all_files_then_skips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            contents = {
                "student_ldm": b"student-bytes",
                "sr_vocoder": b"vocoder-bytes",
                "vae": b"vae-bytes",
            }
            files = [
                {
                    "name": name,
                    "file": f"cache/flashsr/{name}.pth",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source": f"https://example.invalid/{name}.pth",
                }
                for name, payload in contents.items()
            ]
            _write_manifest(
                model_dir,
                [{"id": "flashsr", "backend": "flashsr", "file": "cache/flashsr", "files": files}],
            )

            call_log: list[str] = []

            def fake_download(source: str, part_path: Path, *args: object, **kwargs: object) -> list[Path]:
                name = source.rsplit("/", 1)[-1].replace(".pth", "")
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_bytes(contents[name])
                call_log.append(name)
                return [part_path]

            with patch("audio_ai.model_manager._download_parts", side_effect=fake_download):
                installed = install_model(model_dir, "flashsr")

            self.assertEqual(len(installed), 3)
            self.assertEqual(set(call_log), set(contents))
            for name, payload in contents.items():
                self.assertEqual(
                    (model_dir / f"cache/flashsr/{name}.pth").read_bytes(), payload
                )

            # 第二次运行：文件已存在，应跳过下载
            with patch("audio_ai.model_manager._download_parts", side_effect=fake_download) as second:
                install_model(model_dir, "flashsr")
            second.assert_not_called()

    def _write_deferred_entry(self, model_dir: Path) -> None:
        """延迟校验条目：sha256 为哨兵、size 为 0（无法预先锁定校验值时）。

        HiFi-GAN 的 jik876 权重在构建环境不可达，只能信任来源安装，故该路径
        必须可用且不能因「size 0」被判非法或反复重下。
        """
        _write_manifest(
            model_dir,
            [
                {
                    "id": "hifigan-48k",
                    "backend": "hifigan",
                    "file": "cache/hifigan/generator",
                    "size_bytes": 0,
                    "sha256": "deferred",
                    "source": "https://example.invalid/generator",
                }
            ],
        )

    def test_install_model_accepts_deferred_hash_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            target = model_dir / "cache" / "hifigan" / "generator"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"generator-bytes")
            self._write_deferred_entry(model_dir)

            entry = read_model_entry(model_dir, "hifigan-48k")
            self.assertEqual(entry["artifacts"][0]["sha256"], "deferred")

            # 已存在文件：应直接跳过，绝不重新下载（哨兵无法通过哈希校验）
            with patch(
                "audio_ai.model_manager._download_parts",
                side_effect=AssertionError("deferred entry must not re-download"),
            ):
                installed = install_model(model_dir, "hifigan-48k")
            self.assertEqual([path.name for path in installed], ["generator"])

    def test_install_model_downloads_deferred_entry_without_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self._write_deferred_entry(model_dir)

            def fake_download(
                source: str, part_path: Path, *args: object, **kwargs: object
            ) -> list[Path]:
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_bytes(b"downloaded-generator")  # 与清单哈希无关
                return [part_path]

            with patch("audio_ai.model_manager._download_parts", side_effect=fake_download):
                installed = install_model(model_dir, "hifigan-48k")

            self.assertEqual(
                (model_dir / "cache/hifigan/generator").read_bytes(),
                b"downloaded-generator",
            )
            self.assertEqual([path.name for path in installed], ["generator"])

    def test_install_model_resumes_after_partial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            contents = {
                "student_ldm": b"student-bytes",
                "sr_vocoder": b"vocoder-bytes",
                "vae": b"vae-bytes",
            }
            files = [
                {
                    "name": name,
                    "file": f"cache/flashsr/{name}.pth",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source": f"https://example.invalid/{name}.pth",
                }
                for name, payload in contents.items()
            ]
            _write_manifest(
                model_dir,
                [{"id": "flashsr", "backend": "flashsr", "file": "cache/flashsr", "files": files}],
            )

            downloaded: list[str] = []

            def fake_download(source: str, part_path: Path, *args: object, **kwargs: object) -> list[Path]:
                name = source.rsplit("/", 1)[-1].replace(".pth", "")
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_bytes(contents[name])
                downloaded.append(name)
                return [part_path]

            with patch("audio_ai.model_manager._download_parts", side_effect=fake_download):
                install_model(model_dir, "flashsr")
            self.assertEqual(set(downloaded), set(contents))

            # 删掉一个文件，重跑：只应重新下载被删的那个
            (model_dir / "cache/flashsr/student_ldm.pth").unlink()
            downloaded.clear()
            with patch("audio_ai.model_manager._download_parts", side_effect=fake_download):
                install_model(model_dir, "flashsr")
            self.assertEqual(downloaded, ["student_ldm"])

    def test_install_model_rejects_path_escape_in_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            _write_manifest(
                model_dir,
                [
                    {
                        "id": "flashsr",
                        "backend": "flashsr",
                        "file": "cache/flashsr",
                        "files": [
                            {
                                "name": "vae",
                                "file": "../escaped.pth",
                                "size_bytes": 1,
                                "sha256": "0" * 64,
                                "source": "https://example.invalid/vae.pth",
                            }
                        ],
                    }
                ],
            )
            with self.assertRaisesRegex(ModelInstallError, "escapes model directory"):
                install_model(model_dir, "flashsr")

    def test_verify_file_checks_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            self.assertTrue(verify_file(path, 5, digest))
            self.assertFalse(verify_file(path, 4, digest))
            self.assertFalse(verify_file(path, 5, "0" * 64))

    def test_install_model_keeps_bad_part_and_does_not_replace_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            cache_dir = model_dir / "cache"
            cache_dir.mkdir()
            target = cache_dir / "model.bin"
            target.write_bytes(b"old")
            model_dir.joinpath("manifest.json").write_text(
                '{"models":[{"id":"test","file":"cache/model.bin",'
                '"source":"https://example.invalid/model.bin",'
                '"size_bytes":5,"sha256":"'
                + hashlib.sha256(b"right").hexdigest()
                + '"}]}',
                encoding="utf-8",
            )
            part = target.with_name("model.bin.part")
            with patch("audio_ai.model_manager._download_parts") as download:
                def write_bad_part(*args: object, **kwargs: object) -> None:
                    part.write_bytes(b"wrong")
                    return [part]

                download.side_effect = write_bad_part
                with self.assertRaisesRegex(ModelInstallError, "hash mismatch"):
                    install_model(model_dir, "test")

            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(part.read_bytes(), b"wrong")


if __name__ == "__main__":
    unittest.main()
