# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Offline download integrity tests; only the network transport is replaced."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from examples.task1_qwen35 import download_model as module


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name)
        self.content = b"fixed test model data\n"
        self.entry = {
            "Path": "model.safetensors",
            "Revision": "a" * 40,
            "Sha256": hashlib.sha256(self.content).hexdigest(),
            "Size": len(self.content),
        }
        self.target = self.destination / self.entry["Path"]
        self.partial = self.target.with_name(self.target.name + ".partial")

    def test_verified_existing_never_downloaded(self):
        self.target.write_bytes(self.content)
        with patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=AssertionError("unexpected network"),
        ):
            module.download_file(self.entry, self.destination)
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assertFalse(self.partial.exists())

    def test_success_replaces_bad_file_after_verification(self):
        self.target.write_bytes(b"bad old file")

        def open_official(url, timeout):
            parsed = urlparse(url)
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.netloc, "modelscope.cn")
            self.assertEqual(parsed.path, "/api/v1/models/Qwen/Qwen3.5-0.8B/repo")
            self.assertEqual(
                parse_qs(parsed.query),
                {"Revision": ["a" * 40], "FilePath": ["model.safetensors"]},
            )
            self.assertEqual(self.target.read_bytes(), b"bad old file")
            return io.BytesIO(self.content)

        with patch.object(module.urllib.request, "urlopen", side_effect=open_official):
            module.download_file(self.entry, self.destination)
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assertFalse(self.partial.exists())

    def test_bad_hash_preserves_target_and_partial(self):
        self.target.write_bytes(b"old target")
        bad = b"new corrupt response"
        with patch.object(
            module.urllib.request, "urlopen", return_value=io.BytesIO(bad)
        ):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                module.download_file(self.entry, self.destination)
        self.assertEqual(self.target.read_bytes(), b"old target")
        self.assertEqual(self.partial.read_bytes(), bad)

    def test_bad_size_is_rejected_even_with_matching_hash(self):
        entry = dict(self.entry, Size=len(self.content) + 1)
        with patch.object(
            module.urllib.request, "urlopen", return_value=io.BytesIO(self.content)
        ):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                module.download_file(entry, self.destination)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.partial.read_bytes(), self.content)

    def test_interrupted_stream_preserves_target_and_partial(self):
        class Interrupted(io.BytesIO):
            def read(self, n=-1):
                if self.tell():
                    raise ConnectionError("interrupted")
                return super().read(4)

        self.target.write_bytes(b"old target")
        with patch.object(
            module.urllib.request, "urlopen", return_value=Interrupted(self.content)
        ):
            with self.assertRaises(ConnectionError):
                module.download_file(self.entry, self.destination)
        self.assertEqual(self.target.read_bytes(), b"old target")
        self.assertEqual(self.partial.read_bytes(), self.content[:4])

    def test_partial_symlink_cannot_overwrite_file(self):
        other = self.destination / "good"
        other.write_bytes(b"preserve")
        self.partial.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlink"):
            module.download_file(self.entry, self.destination)
        self.assertEqual(other.read_bytes(), b"preserve")

    def test_valid_manifest_round_trip_downloads_verified_files(self):
        manifest_path = self.destination / "manifest.json"
        contents = {
            "model.safetensors": self.content,
            "config.json": b'{"model_type": "qwen3_5"}\n',
        }
        entries = [
            {
                "Path": name,
                "Revision": "a" * 40,
                "Sha256": hashlib.sha256(content).hexdigest(),
                "Size": len(content),
            }
            for name, content in contents.items()
        ]
        manifest_path.write_text(json.dumps(entries), encoding="utf-8")
        loaded = module.load_manifest(manifest_path)
        self.assertEqual(loaded, entries)

        def open_pinned_file(url, timeout):
            query = parse_qs(urlparse(url).query)
            self.assertEqual(query["Revision"], ["a" * 40])
            return io.BytesIO(contents[query["FilePath"][0]])

        with patch.object(
            module.urllib.request, "urlopen", side_effect=open_pinned_file
        ):
            for entry in loaded:
                module.download_file(entry, self.destination)
        for name, content in contents.items():
            self.assertEqual((self.destination / name).read_bytes(), content)
            self.assertFalse((self.destination / f"{name}.partial").exists())

    def test_manifest_rejects_unsafe_or_unpinned_inputs(self):
        path = self.destination / "manifest.json"
        bad_entries = [
            dict(self.entry, Path="../escape"),
            dict(self.entry, Revision="main"),
            dict(self.entry, Sha256="0"),
            dict(self.entry, Size=-1),
            dict(self.entry, Size=True),
            dict(self.entry, Path="x.partial"),
        ]
        for entry in bad_entries:
            with self.subTest(entry=entry):
                path.write_text(json.dumps([entry]))
                with self.assertRaises(ValueError):
                    module.load_manifest(path)
        path.write_text(json.dumps([self.entry, self.entry]))
        with self.assertRaises(ValueError):
            module.load_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
