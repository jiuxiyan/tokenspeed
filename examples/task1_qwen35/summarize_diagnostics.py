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

"""Aggregate worker measurements without mixing profiler and baseline timings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from examples.task1_qwen35.common import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in sorted((args.run / "worker").glob("*/*.json")):
        item = json.loads(path.read_text())
        groups.setdefault((path.parent.name, item["phase"]), []).append(item)
    if not groups:
        raise RuntimeError("no worker observations found")
    summary = []
    for (label, phase), samples in groups.items():
        durations = [s["model_forward_ms"] for s in samples if not s["profiled"]]
        summary.append(
            {
                "label": label,
                "phase": phase,
                "forwards": len(samples),
                "unprofiled_model_forward_ms_median": (
                    statistics.median(durations) if durations else None
                ),
                "unprofiled_model_forward_ms_min": (
                    min(durations) if durations else None
                ),
                "unprofiled_model_forward_ms_max": (
                    max(durations) if durations else None
                ),
                "profiled_forwards": sum(s["profiled"] for s in samples),
                "peak_allocated_bytes": max(s["max_allocated"] for s in samples),
                "peak_reserved_bytes": max(s["max_reserved"] for s in samples),
            }
        )
    save_json(
        args.output,
        {
            "scope": "ModelRunner.forward on the actual worker; pre/post NPU synchronization; memory peaks reset at each model forward. Excludes scheduler and sampling; diagnostic synchronization can affect overlap.",
            "groups": summary,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
