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

"""CPU tests for metric definitions, trajectory checks and parameter coverage."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from examples.task1_qwen35.compare_hf import (
    compare_logits,
    select_snapshots,
    validate_snapshot,
)
from examples.task1_qwen35.worker_observer import weight_audit


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires CPU PyTorch; no NPU needed"
)
class NumericalBookkeepingTests(unittest.TestCase):
    def test_full_vocabulary_metrics_have_reference_l2_denominator(self):
        import torch

        result = compare_logits(
            torch.tensor([1.0, 2.0, 4.0]), torch.tensor([1.0, 2.0, 3.0])
        )
        self.assertAlmostEqual(result["relative_l2"], 1 / math.sqrt(14), places=6)
        self.assertAlmostEqual(result["rmse"], 1 / math.sqrt(3), places=6)
        self.assertAlmostEqual(result["cosine"], 17 / math.sqrt(21 * 14), places=6)
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["reference_top1_margin"], 1)
        self.assertTrue(result["top1_agrees"])
        self.assertEqual(result["vocabulary_size"], 3)

    def test_nonfinite_logits_are_not_masked(self):
        import torch

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                compare_logits(torch.tensor([bad, 1.0]), torch.tensor([0.0, 1.0]))
        with self.assertRaisesRegex(AssertionError, "vocabulary"):
            compare_logits(torch.ones(3), torch.ones(2))

    def test_snapshot_must_match_input_position_phase_and_greedy_token(self):
        import torch

        fixture = {"input_ids": [10, 11], "output_ids": [2, 2]}
        saved = {
            "input_ids": torch.tensor([2]),
            "positions": torch.tensor([[2], [2], [2]]),
            "phase": "decode",
            "logits": torch.tensor([[0.0, 1.0, 3.0]]),
        }
        validate_snapshot(saved, fixture, 1, 3)
        for key, bad in (
            ("input_ids", torch.tensor([1])),
            ("positions", torch.tensor([[3], [3], [3]])),
            ("phase", "extend"),
            ("logits", torch.tensor([[0.0, 4.0, 3.0]])),
        ):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_snapshot({**saved, key: bad}, fixture, 1, 3)

    def test_only_known_previous_request_extra_decode_is_excluded(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "B.json").write_text(
                json.dumps({"input_ids": [5, 6], "output_ids": [8, 9]})
            )
            snapshots = root / "worker" / "A2"
            snapshots.mkdir(parents=True)
            extra = {
                "input_ids": torch.tensor([9]),
                "positions": torch.tensor([[3], [3], [3]]),
                "phase": "decode",
                "logits": torch.zeros(1, 3),
            }
            torch.save(extra, snapshots / "10_000.pt")
            torch.save({"phase": "extend"}, snapshots / "10_001.pt")
            torch.save({"phase": "decode"}, snapshots / "10_002.pt")
            selected, excluded = select_snapshots(
                root, "A2", {"generation_tokens": 2}, "B"
            )
            self.assertEqual(
                [path.name for path in selected], ["10_001.pt", "10_002.pt"]
            )
            self.assertEqual(excluded[0]["expected_previous_last_output_id"], 9)
            self.assertEqual(excluded[0]["expected_previous_next_position"], 3)
            extra["positions"] = torch.tensor([4])
            torch.save(extra, snapshots / "10_000.pt")
            with self.assertRaisesRegex(AssertionError, "unrecognized"):
                select_snapshots(root, "A2", {"generation_tokens": 2}, "B")

    def test_tied_head_storage_is_covered_but_unloaded_parameter_is_missing(self):
        import torch

        model = torch.nn.Module()
        model.add_module("embedding", torch.nn.Embedding(3, 2))
        model.add_module("lm_head", torch.nn.Linear(2, 3, bias=False))
        model.lm_head.weight = model.embedding.weight
        result = weight_audit(model, [], {"embedding.weight"})
        self.assertTrue(result["all_text_parameters_covered"])
        self.assertEqual(
            result["text_parameters_covered_by_shared_storage"],
            {"lm_head.weight": ["embedding.weight"]},
        )
        model.add_module("missing", torch.nn.Linear(2, 2, bias=False))
        self.assertEqual(
            weight_audit(model, [], {"embedding.weight"})["missing_text_parameters"],
            ["missing.weight"],
        )

    def test_distinct_views_of_one_storage_are_not_tied_parameters(self):
        import torch

        model = torch.nn.Module()
        base = torch.zeros(4)
        model.register_parameter("a", torch.nn.Parameter(base[:2]))
        model.register_parameter("b", torch.nn.Parameter(base[2:]))
        result = weight_audit(model, [], {"a"})
        self.assertFalse(result["all_text_parameters_covered"])
        self.assertEqual(result["missing_text_parameters"], ["b"])


if __name__ == "__main__":
    unittest.main()
