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

"""Host-only fixtures for the confirmed installed profiler export schema."""

import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from examples.task1_qwen35.analyze_profile import (
    KERNEL_FIELDS,
    OPERATOR_FIELDS,
    interval_summary,
    kernel_summary,
    number,
    operator_summary,
    trace_summary,
)


def write_fixture_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class ProfileAnalysisTests(unittest.TestCase):
    def test_overlapping_durations_are_unioned_and_gaps_preserved(self):
        result = interval_summary(
            [
                (Decimal(0), Decimal(10)),
                (Decimal(5), Decimal(15)),
                (Decimal(20), Decimal(25)),
            ]
        )
        self.assertEqual(result["summed_duration_us"], 25)
        self.assertEqual(result["union_busy_us"], 20)
        self.assertEqual(result["span_us"], 25)
        self.assertEqual(result["gap_us"], 5)
        self.assertEqual(result["busy_fraction_within_selected_event_span"], 0.8)
        self.assertEqual(result["overlap_multiplicity_us"], 5)

    def test_large_timestamps_preserve_submicrosecond_intervals(self):
        start = number("1700000000000000.001")
        result = interval_summary([(start, start + number("0.003"))])
        self.assertEqual(result["union_busy_us"], 0.003)
        self.assertEqual(result["span_start_us"], "1700000000000000.001")
        with self.assertRaises(ValueError):
            interval_summary([(number(2), number(1))])
        with self.assertRaises(ValueError):
            number("NaN")

    def test_kernel_type_groups_keep_exported_core_and_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_details.csv"
            write_fixture_csv(
                path,
                KERNEL_FIELDS,
                [
                    ["0", "matmul_1", "MatMul", "AI_CORE", 0, 10],
                    ["0", "matmul_2", "MatMul", "AI_CORE", 5, 10],
                    ["0", "copy", "Memcpy", "AI_CPU", 20, 3],
                ],
            )
            result = kernel_summary(path)
            self.assertEqual(result["types"][0]["count"], 2)
            self.assertEqual(result["types"][0]["summed_device_duration_us"], 20)
            self.assertEqual(result["devices"]["0"]["union_busy_us"], 18)
            self.assertEqual(result["devices"]["0"]["summed_duration_us"], 23)

    def test_unknown_csv_columns_fail_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_details.csv"
            write_fixture_csv(path, ["Name", "duration_ms"], [["x", 1]])
            with self.assertRaisesRegex(ValueError, "schema"):
                kernel_summary(path)

    def test_operator_self_and_total_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operator_details.csv"
            write_fixture_csv(
                path, OPERATOR_FIELDS, [["parent", 1, 8, 2, 10], ["child", 3, 3, 8, 8]]
            )
            result = operator_summary(path)
            self.assertEqual(result["operators"][0]["name"], "child")
            self.assertEqual(result["operators"][1]["Device Total Duration(us)"], 10)
            self.assertIn("not additive", result["scope"])

    def test_trace_selects_only_uniquely_linked_device_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            events = [
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": 7,
                    "args": {"name": "fixture device"},
                },
                {"ph": "X", "pid": 1, "tid": 1, "name": "CPU", "ts": 0, "dur": 100},
                {
                    "ph": "X",
                    "pid": 7,
                    "tid": 2,
                    "name": "kernel_a",
                    "ts": 20,
                    "dur": 20,
                    "cat": "fixture",
                    "args": {"Task Type": "AI_CORE"},
                },
                {
                    "ph": "X",
                    "pid": 7,
                    "tid": 3,
                    "name": "kernel_b",
                    "ts": 30,
                    "dur": 20,
                    "cat": "fixture",
                },
                {
                    "ph": "X",
                    "pid": 7,
                    "tid": 3,
                    "name": "unlinked",
                    "ts": 70,
                    "dur": 10,
                },
            ]
            for flow_id, tid, ts in ((1, 2, 20), (2, 3, 30), (3, 4, 90)):
                events += [
                    {
                        "ph": "s",
                        "cat": "HostToDevice",
                        "id": flow_id,
                        "pid": 1,
                        "tid": 1,
                        "ts": 5,
                    },
                    {
                        "ph": "f",
                        "cat": "HostToDevice",
                        "id": flow_id,
                        "pid": 7,
                        "tid": tid,
                        "ts": ts,
                    },
                ]
            path.write_text(json.dumps(events))
            result = trace_summary(path, selection="host-to-device", device_pids=[])
            self.assertEqual(result["whole_trace_complete_event_span_us"], 100)
            self.assertEqual(result["selected"]["summed_duration_us"], 40)
            self.assertEqual(result["selected"]["union_busy_us"], 30)
            self.assertEqual(
                result["selected_busy_fraction_within_whole_trace_span"], 0.3
            )
            self.assertEqual(len(result["unmatched_flows"]), 1)
            explicit = trace_summary(path, selection="explicit-pid", device_pids=["7"])
            self.assertEqual(explicit["selected"]["union_busy_us"], 40)
            with self.assertRaises(ValueError):
                trace_summary(path, selection="explicit-pid", device_pids=["999"])


if __name__ == "__main__":
    unittest.main()
