"""Tests for explicit model installation without network access."""

from __future__ import annotations

import hashlib
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
            with patch("audio_ai.model_manager._download_part") as download:
                def write_bad_part(*args: object, **kwargs: object) -> None:
                    part.write_bytes(b"wrong")

                download.side_effect = write_bad_part
                with self.assertRaisesRegex(ModelInstallError, "hash mismatch"):
                    install_model(model_dir, "test")

            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(part.read_bytes(), b"wrong")


if __name__ == "__main__":
    unittest.main()
