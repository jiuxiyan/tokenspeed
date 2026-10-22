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

"""Check token-count and latency denominators without accelerator dependencies."""

import json
import tempfile
import unittest
from pathlib import Path

from examples.task1_qwen35.common import (
    code_identity,
    ids_hash,
    stream_timings,
    timing_summary,
)
from examples.task1_qwen35.summarize_results import markdown_report, read_benchmark


class TimingTests(unittest.TestCase):
    def test_first_token_is_prefill_and_empty_final_frame_is_not_decode(self):
        events = [
            {"time": 1.1, "new_ids": [10]},
            {"time": 1.3, "new_ids": [11, 12]},
            {"time": 1.4, "new_ids": []},
        ]
        result = stream_timings(events, 1.0, 1.5, 3)
        self.assertAlmostEqual(result["ttft_ms"], 100)
        self.assertAlmostEqual(result["tpot_ms"], 100)
        self.assertEqual(result["decode_steps"], 2)
        self.assertAlmostEqual(result["total_ms"], 500)
        self.assertEqual(result["output_ids"], [10, 11, 12])

    def test_coalesced_first_frame_does_not_claim_exact_ttft(self):
        result = stream_timings([{"time": 2.0, "new_ids": [10, 11]}], 1.0, 2.1, 2)
        self.assertFalse(result["ttft_is_exact"])
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["tpot_ms"])
        self.assertEqual(timing_summary([result])["ttft_ms"]["n"], 0)

    def test_one_generated_token_has_no_decode_step(self):
        result = stream_timings([{"time": 2.0, "new_ids": [10]}], 1.0, 2.1, 1)
        self.assertEqual(result["decode_steps"], 0)
        self.assertIsNone(result["tpot_ms"])

    def test_early_eos_or_missing_tokens_is_an_error(self):
        with self.assertRaises(AssertionError):
            stream_timings([{"time": 2.0, "new_ids": [10]}], 1.0, 2.1, 2)


class SourceIdentityTests(unittest.TestCase):
    def test_archive_revision_and_actual_code_changes_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "0123456789abcdef0123456789abcdef01234567"
            (root / "BASE_COMMIT").write_text(revision + "\n")
            source = root / "python" / "model.py"
            source.parent.mkdir()
            source.write_text("value = 1\n")
            first = code_identity(root)
            self.assertEqual(first["commit"], revision)
            self.assertEqual(first["commit_source"], "BASE_COMMIT")
            self.assertIsNone(first["working_tree_status"])
            self.assertEqual(first["source_file_count"], 1)
            source.write_text("value = 2\n")
            second = code_identity(root)
            self.assertNotEqual(
                first["source_manifest_sha256"], second["source_manifest_sha256"]
            )

    def test_invalid_archive_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BASE_COMMIT").write_text("unknown\n")
            with self.assertRaises(ValueError):
                code_identity(root)


class SummaryTests(unittest.TestCase):
    def test_warmup_is_excluded_and_incomplete_matrix_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for index, latency in enumerate((1000.0, 100.0, 200.0)):
                label = f"request_{index}"
                cases.append({"label": label, "warmup": index == 0})
                sample = {
                    "label": label,
                    "input_ids": [1, 2],
                    "input_tokens": 2,
                    "input_sha256": ids_hash([1, 2]),
                    "output_ids": [3, 4, 5],
                    "fixed_length": True,
                    "generation_tokens": 3,
                    "generation_limit": 3,
                    "decode_steps": 2,
                    "meta_info": {"cached_tokens": 0},
                    "ttft_is_exact": True,
                    "ttft_ms": latency,
                    "tpot_ms": 2.0,
                    "total_ms": latency + 4,
                }
                (root / f"{label}.json").write_text(json.dumps(sample))
            manifest = {
                "mode": "benchmark",
                "completed": True,
                "warmups": 1,
                "repeats": 2,
                "cases": cases,
                "environment": {},
                "engine_config": {},
                "measurement": {},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            result = read_benchmark(root, allow_partial=False)
            self.assertEqual(result["rows"][0]["ttft_ms"]["median"], 150.0)
            self.assertEqual(result["rows"][0]["ttft_ms"]["n"], 2)
            self.assertEqual(result["rows"][0]["warmups"], 1)
            self.assertFalse(result["protocol_matrix_complete"])
            rendered = markdown_report({"benchmark": result, "diagnostics": []})
            self.assertIn("150.000 [100.000, 200.000]", rendered)
            manifest["completed"] = False
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_benchmark(root, allow_partial=False)


if __name__ == "__main__":
    unittest.main()
